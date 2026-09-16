"""LMI-based polytopic state-feedback synthesis for the TP-GDAR model.

The fitted model is lag-K; we first lift it to a first-order block-companion
system on the stacked state ``xi(t) = [x(t); x(t-1); ...; x(t-K+1)]``:

    xi(t+1) = A_c(p) xi(t) + B_c u(t),   A_c(p) = sum_v lambda_v(p) A_c^v

Then synthesize a single state-feedback gain ``F`` (``u = F xi``) that
quadratically stabilizes the *whole polytope* via the discrete-time LMIs

    Q > 0,   [[ Q,            (A_c^v Q + B_c Y)^T ],
              [ A_c^v Q + B_c Y,   Q            ]] > 0   for every vertex v,

with ``F = Y Q^{-1}``.  A less-conservative slack-variable (de Oliveira) variant
is also provided.  Requires cvxpy with an SDP solver (SCS or Clarabel).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import cvxpy as cp
    _HAVE_CVXPY = True
except Exception:  # pragma: no cover
    _HAVE_CVXPY = False


def companion_form(A_list, B):
    """Lift lag-K ``[A_0..A_{K-1}]`` + input ``B`` to first-order companion form.

    Returns
    -------
    A_c : (K*N, K*N)
    B_c : (K*N, M)
    """
    A_list = [np.asarray(A, dtype=float) for A in A_list]
    K = len(A_list)
    N = A_list[0].shape[0]
    B = np.asarray(B, dtype=float)
    M = B.shape[1] if B.ndim == 2 else 0
    A_c = np.zeros((K * N, K * N))
    A_c[:N, :] = np.hstack(A_list)          # top block row = the lag matrices
    if K > 1:
        A_c[N:, :-N] = np.eye((K - 1) * N)  # shift register
    B_c = np.zeros((K * N, M))
    if M:
        B_c[:N, :] = B
    return A_c, B_c


def _vertices_to_companion(vertex_A, B):
    """vertex_A: (V, N, N, K) or list of [A_0..A_{K-1}]; returns companion vertices."""
    comp = []
    for v in vertex_A:
        if isinstance(v, np.ndarray) and v.ndim == 3:
            A_list = [v[..., k] for k in range(v.shape[-1])]  # (N,N,K) -> list
        else:
            A_list = list(v)
        Ac, Bc = companion_form(A_list, B)
        comp.append(Ac)
    return comp, Bc


@dataclass
class SynthesisResult:
    F: np.ndarray | None
    Q: np.ndarray | None
    feasible: bool
    method: str
    max_rho: float          # worst closed-loop spectral radius over vertices
    solver: str = ""
    status: str = ""


def synthesize_quadratic_feedback(vertex_A, B, margin=1e-6, solver=None,
                                   normalize=True):
    """Quadratic-stability state feedback over the polytope.

    Parameters
    ----------
    vertex_A : sequence of vertex systems, each either an ``(N, N, K)`` array
        (lags in the last axis) or a list ``[A_0, ..., A_{K-1}]``.
    B : (N, M) input matrix (shared across vertices).
    margin : float, LMI strict-positivity margin.

    Returns
    -------
    SynthesisResult
    """
    if not _HAVE_CVXPY:
        raise RuntimeError("cvxpy is required for LMI synthesis")
    comp, Bc = _vertices_to_companion(vertex_A, B)
    nx = comp[0].shape[0]
    m = Bc.shape[1]

    Q = cp.Variable((nx, nx), symmetric=True)
    Y = cp.Variable((m, nx))
    cons = [Q >> (1.0 if normalize else margin) * np.eye(nx)]
    for Ac in comp:
        top = cp.hstack([Q, (Ac @ Q + Bc @ Y).T])
        bot = cp.hstack([Ac @ Q + Bc @ Y, Q])
        lmi = cp.vstack([top, bot])
        cons.append(lmi >> margin * np.eye(2 * nx))
    prob = cp.Problem(cp.Minimize(cp.trace(Q)), cons)
    _solve(prob, solver)

    return _finish(prob, Q, Y, comp, Bc, "quadratic")


def synthesize_slack_feedback(vertex_A, B, margin=1e-6, solver=None):
    """Slack-variable (de Oliveira) robust state feedback.

    Introduces a non-symmetric slack ``G`` and a common ``P``; often feasible
    where the plain quadratic LMI is not, at the cost of more variables.
    """
    if not _HAVE_CVXPY:
        raise RuntimeError("cvxpy is required for LMI synthesis")
    comp, Bc = _vertices_to_companion(vertex_A, B)
    nx = comp[0].shape[0]
    m = Bc.shape[1]

    P = cp.Variable((nx, nx), symmetric=True)
    G = cp.Variable((nx, nx))
    Y = cp.Variable((m, nx))
    cons = [P >> np.eye(nx)]
    for Ac in comp:
        AG = Ac @ G + Bc @ Y
        top = cp.hstack([G + G.T - P, AG.T])
        bot = cp.hstack([AG, P])
        cons.append(cp.vstack([top, bot]) >> margin * np.eye(2 * nx))
    prob = cp.Problem(cp.Minimize(cp.trace(P)), cons)
    _solve(prob, solver)

    if prob.status not in ("optimal", "optimal_inaccurate") or G.value is None:
        return SynthesisResult(F=None, Q=None, feasible=False, method="slack",
                               max_rho=np.inf, solver=str(solver), status=prob.status)
    F = Y.value @ np.linalg.inv(G.value)
    max_rho = closed_loop_max_rho(comp, Bc, F)
    return SynthesisResult(F=F, Q=P.value, feasible=max_rho < 1.0,
                           method="slack", max_rho=max_rho,
                           solver=str(solver), status=prob.status)


def _solve(prob, solver):
    if solver is not None:
        prob.solve(solver=solver)
        return
    for s in ("CLARABEL", "SCS"):
        try:
            prob.solve(solver=s)
            if prob.status in ("optimal", "optimal_inaccurate"):
                return
        except Exception:
            continue


def _finish(prob, Q, Y, comp, Bc, method):
    if prob.status not in ("optimal", "optimal_inaccurate") or Q.value is None:
        return SynthesisResult(F=None, Q=None, feasible=False, method=method,
                               max_rho=np.inf, status=prob.status)
    Qv = Q.value
    F = Y.value @ np.linalg.inv(Qv)
    max_rho = closed_loop_max_rho(comp, Bc, F)
    return SynthesisResult(F=F, Q=Qv, feasible=max_rho < 1.0, method=method,
                           max_rho=max_rho, status=prob.status)


def closed_loop_max_rho(comp_vertices, Bc, F):
    """Worst-case closed-loop spectral radius over the polytope vertices."""
    rho = 0.0
    for Ac in comp_vertices:
        eig = np.linalg.eigvals(Ac + Bc @ F)
        rho = max(rho, float(np.max(np.abs(eig))))
    return rho


def is_schur_stable(A, tol=1.0):
    """True if all eigenvalues of ``A`` lie strictly inside the unit disk."""
    return float(np.max(np.abs(np.linalg.eigvals(A)))) < tol
