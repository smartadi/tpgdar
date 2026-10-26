"""Wilson-Cowan oscillator network on a graph.

Each node carries an excitatory (E) and inhibitory (I) population.  Nodes are
coupled through the graph adjacency on their excitatory populations, with a
global coupling gain ``g`` that can be modulated by a scheduling variable ``p``.
This is the nonlinear *plant* we fit GDAR models to and, later, close the control
loop on (never on the fitted model itself).

Discrete Euler integration of the standard rate model::

    tau_E E'_i = -E_i + f(c_EE E_i - c_EI I_i + g(p) sum_j W_ij E_j + P_E + s_i(t))
    tau_I I'_i = -I_i + f(c_IE E_i - c_II I_i + P_I)

with ``f`` a logistic sigmoid.  The observed channel signal is ``E`` (optionally
mean-subtracted), matching what a recording electrode sees.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def sigmoid(x, a=1.0, theta=0.0):
    return 1.0 / (1.0 + np.exp(-a * (x - theta)))


@dataclass
class WCParams:
    # Local E-I connectivity (Wilson-Cowan classic oscillatory regime).
    c_ee: float = 16.0
    c_ei: float = 12.0
    c_ie: float = 15.0
    c_ii: float = 3.0
    # Time constants (ms).
    tau_e: float = 8.0
    tau_i: float = 8.0
    # Background drive.
    p_e: float = 1.25
    p_i: float = 0.0
    # Sigmoid shape.
    a_e: float = 1.3
    theta_e: float = 4.0
    a_i: float = 2.0
    theta_i: float = 3.7
    # Base network coupling gain (scaled by the scheduling function).
    g0: float = 0.9
    dt: float = 1.0  # ms


@dataclass
class WCNetwork:
    """Wilson-Cowan network on a given adjacency matrix."""

    adjacency: np.ndarray
    params: WCParams = field(default_factory=WCParams)
    normalize_coupling: bool = True

    def __post_init__(self):
        self.n = self.adjacency.shape[0]
        W = self.adjacency.astype(float)
        if self.normalize_coupling:
            deg = W.sum(axis=1, keepdims=True)
            deg[deg == 0] = 1.0
            W = W / deg  # row-stochastic: coupling is an average over neighbours
        self.W = W

    def step(self, E, I, g, drive, noise_std=0.0, rng=None):
        """Advance one Euler step. ``drive`` is a per-channel additive E input."""
        pr = self.params
        net_in = pr.c_ee * E - pr.c_ei * I + g * (self.W @ E) + pr.p_e + drive
        inh_in = pr.c_ie * E - pr.c_ii * I + pr.p_i
        E_new = E + (pr.dt / pr.tau_e) * (-E + sigmoid(net_in, pr.a_e, pr.theta_e))
        I_new = I + (pr.dt / pr.tau_i) * (-I + sigmoid(inh_in, pr.a_i, pr.theta_i))
        if noise_std:
            rng = np.random.default_rng(rng)
            E_new = E_new + noise_std * rng.standard_normal(E.shape)
            I_new = I_new + noise_std * rng.standard_normal(I.shape)
        return E_new, I_new

    def coupling_gain(self, p):
        """Scheduling map p -> global coupling gain g(p).

        Affine in ``p`` so that the *linearized* GDAR coefficients are (to first
        order) affine in ``p`` as well -> a rank-2 parameter dependence, i.e. two
        TP vertices for a 1-D schedule.  ``p`` is expected in roughly [0, 1].
        """
        return self.params.g0 * (0.5 + 1.0 * np.asarray(p, dtype=float))

    def simulate(self, n_steps, p=0.5, stim=None, B=None, U=None,
                 noise_std=0.0, rng=None, burn_in=500, x0=None,
                 return_full=False):
        """Integrate the network.

        Parameters
        ----------
        n_steps : int
        p : float or (n_steps + burn_in,) array
            Scheduling variable, constant or time-varying.
        stim : (n_steps + burn_in, N) array, optional
            Direct per-channel additive drive to E (bypasses B/U).
        B : (N, M) array, optional
        U : (n_steps + burn_in, M) array, optional
            Input mapped to channels via ``B`` (added to E drive).
        noise_std : float
            Std of additive Gaussian process noise on E and I.
        burn_in : int
        x0 : (2, N) array, optional
            Initial (E, I). Small random if omitted.
        return_full : bool
            If True also return the inhibitory trace.

        Returns
        -------
        E : (n_steps, N) ndarray   (observed channels)
        (I : (n_steps, N) ndarray  if return_full)
        """
        rng = np.random.default_rng(rng)
        pr = self.params
        total = n_steps + burn_in
        n = self.n

        p_arr = np.broadcast_to(np.asarray(p, dtype=float), (total,)) if np.ndim(p) == 0 \
            else np.asarray(p, dtype=float)
        if p_arr.shape[0] < total:
            raise ValueError("time-varying p must cover n_steps + burn_in")
        g_arr = self.coupling_gain(p_arr)

        drive = np.zeros((total, n))
        if stim is not None:
            drive += np.asarray(stim, dtype=float)[:total]
        if B is not None and U is not None:
            drive += np.asarray(U, dtype=float)[:total] @ np.asarray(B).T

        if x0 is None:
            E = 0.1 + 0.01 * rng.standard_normal(n)
            I = 0.1 + 0.01 * rng.standard_normal(n)
        else:
            E, I = np.array(x0[0], dtype=float), np.array(x0[1], dtype=float)

        out_e = np.zeros((total, n))
        out_i = np.zeros((total, n))
        de_dt = pr.dt / pr.tau_e
        di_dt = pr.dt / pr.tau_i
        # Convention (matches sim/linear.py and GDARModel): x(t+1) = f(x(t), u(t)),
        # so the state written at loop index t is driven by drive[t-1].
        for t in range(total):
            d_prev = drive[t - 1] if t >= 1 else 0.0
            net_in = (pr.c_ee * E - pr.c_ei * I
                      + g_arr[t] * (self.W @ E) + pr.p_e + d_prev)
            inh_in = pr.c_ie * E - pr.c_ii * I + pr.p_i
            E_new = E + de_dt * (-E + sigmoid(net_in, pr.a_e, pr.theta_e))
            I_new = I + di_dt * (-I + sigmoid(inh_in, pr.a_i, pr.theta_i))
            if noise_std:
                E_new += noise_std * rng.standard_normal(n)
                I_new += noise_std * rng.standard_normal(n)
            E, I = E_new, I_new
            out_e[t] = E
            out_i[t] = I

        E_obs = out_e[burn_in:]
        if return_full:
            return E_obs, out_i[burn_in:]   # observed E, ground-truth hidden I
        return E_obs
