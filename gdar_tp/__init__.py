"""GDAR-TP: Graph Diffusion AutoRegression with Tensor-Product vertex structure.

A parameter-varying GDAR model on array graphs, decomposed via TP model
transformation (HOSVD) into vertex LTI systems, for interpretable dynamic
communication modes and LMI-based closed-loop stimulation control.
"""

from . import control, gdar, graph, metrics, qlpv, tp, viz  # noqa: F401
from .gdar import GDARModel, coupling_matrix, edge_flow, fit_gdar
from .graph import GridGraph, grid_graph, kron_sum_laplacian, support_mask
from .qlpv import QLPVModel, fit_grid, fit_grid_2d, fit_windowed
from .tp import Polytope, TPModel, convex_hull, hosvd

__all__ = [
    "graph", "gdar", "qlpv", "tp", "control", "metrics", "viz",
    "GridGraph", "grid_graph", "kron_sum_laplacian", "support_mask",
    "GDARModel", "fit_gdar", "coupling_matrix", "edge_flow",
    "QLPVModel", "fit_grid", "fit_grid_2d", "fit_windowed",
    "TPModel", "Polytope", "hosvd", "convex_hull",
]

__version__ = "0.1.0"
