# Moved to `docs/MATH.md`

On 2026-11-21 this file's content moved into [`docs/MATH.md`](MATH.md), the one math document for the repo, which also covers
the IOS Reach work (Felix's symmetric GLS class, scoring, vertices, single-trial weights). Old section numbers map as follows:

| `MODEL.md` (2026-10-26) | `MATH.md` |
|---|---|
| §1 GDAR at one operating point, ridge | §0, §4.4 |
| §2 TP structure on the stacked operator | §5.2, §12.1 |
| §3a two-stage | §7.4 |
| §3b joint fit by alternating least squares | §6.8 |
| §3c evaluation | §8 |
| §4 polytope and control | §9, §12 |
| §5 preprocessing rules | §1.2 |

Two statements here no longer describe all of the code: the orthonormal-W gauge (§3b) holds for the edge-wise learner only, and
"each model uses its own Σ" (§3c) has exceptions. `MATH.md` §6.8 and §8.6 give the details.
