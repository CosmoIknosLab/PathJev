# Versions of the Stage 1 analysis plan

`docs/stage1_analysis_plan.md` only grows by appending amendments, so each earlier version can be rebuilt from the current file and checked against its hash.

| Version | Frozen (local time) | SHA-256 | How to rebuild it from the current file |
|---|---|---|---|
| v1.0 | 2026-10-17 09:48 (D-025) | `68a18d157e6058d13d66ba336d2571070ed87c5bc67654d2724003360c00aa22` | keep the text before `## Amendments`, then add `## Amendments\n(none)\n` |
| + A1 | 2026-10-20 10:34 (D-028) | `c21d0e5d279cb0ac4541544bfd31f033177cbc476ffb806715bf72a30f4e5ed2` | keep the text before `**A2`, strip trailing newlines, then add one `\n` |
| + A2 | 2026-10-21 19:29 (D-036) | `bfe701859e136f90f9f2af75b11de22b7142eb2b393754f4d9b84903b32df042` | keep the text before `\n\n**A3`, then add one `\n` |
| + A3 | 2026-10-22 14:54 (D-040) | `497508189b8b6971393fde62e8dcff393cf997ed766e6005786b0a84402b91ce` | the current file |

## Timeline

Times come from the progress journal and from sacct.

- **2026-10-19 18:35** — the first test-set numbers are produced, by the pilot (job 5468: 1 seed, 1 epoch, subsets). No hypothesis was tested on the pilot (plan §6).
- **2026-10-20 10:34** — A1 is adopted, after the pilot.
- **2026-10-20 10:42** — the first Phase 4 test job (5757) starts.
- **2026-10-21** — A2 is specified after interim Phase 4 results had been seen, so it is reported as post-hoc.
- **2026-10-22 14:54** — A3 is specified after the full results and an external review had been seen. It is post-hoc and was frozen before any A3 run.

The rebuild rules were checked on 2026-10-22: every rebuilt version reproduces its hash.
