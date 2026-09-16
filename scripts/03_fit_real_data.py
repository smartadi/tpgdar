"""Stage 3 -- real NHP ECoG / Utah recordings.

DEFERRED: we build and validate the full pipeline on synthetic data first
(scripts 01, 02, 05); real data comes later.  This script is the intended entry
point once recordings are wired in through ``gdar_tp.io``.

Planned flow (see gdar_tp/io.py TODO):
  1. load + bandpass + downsample; align to stim events and behavioural labels
  2. build the array GridGraph from electrode geometry
  3. choose scheduling candidates p (stim amplitude, per-band power, reach phase,
     low-dim latent) and fit windowed qLPV -> coefficient tensor
  4. TP decomposition; cross-validate on held-out trials AND held-out conditions
  5. baselines: static GDAR, plain VAR, per-condition GDAR without TP
"""

import sys


def main():
    print(__doc__)
    print("Stage 3 is not yet active -- run scripts 01/02/05 on synthetic data.")
    print("Wire real loaders into gdar_tp/io.py, then implement this script.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
