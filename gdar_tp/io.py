"""Loaders for real ECoG / Utah-array recordings (Stage 3).

Deferred until we move onto brain data.  The intended surface is sketched here so
the pipeline (qlpv -> tp -> control) can be pointed at real recordings without
touching the modeling code: everything downstream consumes ``(X, U, p_series)``
plus a :class:`~gdar_tp.graph.GridGraph`.

TODO(brain-data):
  * ``load_recording(path)`` -> raw channels x time + sample rate + electrode map.
  * bandpass / notch / downsample; align to stim events and behavioural labels.
  * choose scheduling candidates p: stim amplitude, per-band power, reach phase,
    low-dim latent (PCA / LDS).
  * build the array GridGraph from the electrode geometry.
"""

from __future__ import annotations

import numpy as np


def make_synthetic_recording(*args, **kwargs):
    """Placeholder pointing to the synthetic simulators used until Stage 3.

    Use :mod:`gdar_tp.sim.wilson_cowan` / :mod:`gdar_tp.sim.linear` directly for
    now; real-data loaders land here when we start on the ECoG/Utah recordings.
    """
    raise NotImplementedError(
        "Real-data loaders are Stage 3. Use gdar_tp.sim.* for synthetic data."
    )
