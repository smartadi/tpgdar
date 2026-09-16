"""Graph construction for array recordings.

Utah / ECoG arrays live on (roughly) rectangular grids.  This module builds the
grid graph, its Laplacian, the edge list, and the *support mask* ``E ∪ diag`` that
the GDAR fit uses to constrain which channel-to-channel couplings are allowed.

The key structural fact we lean on for the diffusion parameterization is the
Kronecker-sum identity for grid Laplacians::

    L(grid(r, c)) = L(path(r)) ⊕ L(path(c))
                  = L_row ⊗ I_c  +  I_r ⊗ L_col

which is verified in :func:`kron_sum_laplacian` and in ``tests/test_graph.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GridGraph:
    """A rectangular array graph.

    Attributes
    ----------
    rows, cols : int
        Grid dimensions; ``N = rows * cols`` channels.
    adjacency : (N, N) ndarray
        0/1 adjacency (4-neighbour by default).
    laplacian : (N, N) ndarray
        Combinatorial Laplacian ``D - A``.
    edges : list[tuple[int, int]]
        Undirected edge list with ``i < j``.
    """

    rows: int
    cols: int
    adjacency: np.ndarray
    laplacian: np.ndarray
    edges: list[tuple[int, int]]

    @property
    def n(self) -> int:
        return self.rows * self.cols

    def coords(self) -> np.ndarray:
        """(N, 2) integer (row, col) coordinate of each channel (row-major)."""
        rr, cc = np.divmod(np.arange(self.n), self.cols)
        return np.column_stack([rr, cc])


def path_laplacian(n: int) -> np.ndarray:
    """Combinatorial Laplacian of a path (1-D chain) graph of ``n`` nodes."""
    if n == 1:
        return np.zeros((1, 1))
    adj = np.zeros((n, n))
    idx = np.arange(n - 1)
    adj[idx, idx + 1] = 1.0
    adj[idx + 1, idx] = 1.0
    deg = adj.sum(axis=1)
    return np.diag(deg) - adj


def grid_graph(rows: int, cols: int, diagonal: bool = False) -> GridGraph:
    """Build a ``rows x cols`` grid graph.

    Parameters
    ----------
    diagonal : bool
        If True, add 8-neighbour (king-move) diagonal edges as well.
    """
    n = rows * cols

    def node(r, c):
        return r * cols + c

    adj = np.zeros((n, n))
    edges: list[tuple[int, int]] = []
    steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if diagonal:
        steps += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for r in range(rows):
        for c in range(cols):
            i = node(r, c)
            for dr, dc in steps:
                rr, cc = r + dr, c + dc
                if 0 <= rr < rows and 0 <= cc < cols:
                    j = node(rr, cc)
                    adj[i, j] = 1.0
                    if i < j:
                        edges.append((i, j))
    deg = adj.sum(axis=1)
    lap = np.diag(deg) - adj
    return GridGraph(rows=rows, cols=cols, adjacency=adj, laplacian=lap, edges=sorted(set(edges)))


def kron_sum_laplacian(rows: int, cols: int) -> np.ndarray:
    """Grid Laplacian assembled as the Kronecker sum of the two path Laplacians.

    Returns ``L_row ⊗ I_c + I_r ⊗ L_col``, which equals the 4-neighbour grid
    Laplacian from :func:`grid_graph` (see :func:`grid_graph` node ordering:
    row-major, so the row index is the *outer* Kronecker factor).
    """
    lr = path_laplacian(rows)
    lc = path_laplacian(cols)
    return np.kron(lr, np.eye(cols)) + np.kron(np.eye(rows), lc)


def support_mask(adjacency: np.ndarray) -> np.ndarray:
    """Boolean ``(N, N)`` mask of allowed coefficient entries: edges + diagonal."""
    n = adjacency.shape[0]
    mask = adjacency > 0
    mask |= np.eye(n, dtype=bool)
    return mask


def normalized_laplacian(laplacian: np.ndarray) -> np.ndarray:
    """Symmetric normalized Laplacian ``I - D^{-1/2} A D^{-1/2}``.

    Uses the degrees implied by the combinatorial Laplacian's diagonal.
    Isolated nodes (degree 0) are left as ``1`` on the diagonal.
    """
    deg = np.diag(laplacian).copy()
    adj = np.diag(deg) - laplacian
    with np.errstate(divide="ignore"):
        dinv = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    n = laplacian.shape[0]
    norm = np.eye(n) - (dinv[:, None] * adj * dinv[None, :])
    return norm
