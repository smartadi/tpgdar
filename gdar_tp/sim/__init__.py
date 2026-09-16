"""Synthetic plants: linear GDAR generator and the Wilson-Cowan network."""

from . import linear, wilson_cowan  # noqa: F401
from .wilson_cowan import WCNetwork, WCParams
from .closed_loop import run_closed_loop, ClosedLoopResult

__all__ = ["linear", "wilson_cowan", "WCNetwork", "WCParams",
           "run_closed_loop", "ClosedLoopResult"]
