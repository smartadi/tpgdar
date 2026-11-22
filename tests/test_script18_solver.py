"""Script 18's batched simplex solver against the library one (MATH §9.5 face enumeration), for 3 and 4 vertices.

From 2026-11-22 script 18 works on the 4 vertices of the regular hull (MATH §9.6), so its old 3-vertex solver was
replaced by a batched face enumeration for any V.  It must give the library's exact answer on every stack entry.
"""

import importlib
import os
import sys

import numpy as np
import pytest

from gdar_tp.sym_tp import simplex_qp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
s18 = importlib.import_module("18_single_trial_states")


def _obj(Q, b, lam):
    return lam @ Q @ lam - 2 * b @ lam


@pytest.mark.parametrize("V", [3, 4])
def test_batched_solver_matches_the_library(V):
    rng = np.random.default_rng(V)
    n = 300
    B = rng.standard_normal((n, V, V + 2))
    Q = B @ B.transpose(0, 2, 1)
    Q[:40] = Q[:40] * 0 + np.eye(V)[None] * 1e-3 + np.ones((V, V))[None]     # nearly singular along 1
    b = rng.standard_normal((n, V)) * 2
    lam = s18.simplex_qp(Q, b)
    assert lam.min() >= 0 and np.allclose(lam.sum(1), 1)
    for i in range(n):
        ref = simplex_qp(Q[i], b[i])
        assert _obj(Q[i], b[i], lam[i]) <= _obj(Q[i], b[i], ref) + 1e-9 * (1 + abs(_obj(Q[i], b[i], ref)))
