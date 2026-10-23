# Stage 1 analysis plan (pre-registered)

Version 1.0, written 2026-10-17, **before any Stage 1 result exists** (no PathJev-P run, no reportable baseline run on ADHOC). Frozen by recording its sha256 in `docs/DECISIONS.md` (D-025). Any later change is a dated amendment with its reason, listed at the end; changes after results are seen are labelled as such and need the user's approval (CLAUDE.md §3.3, §11).

Scope: Stage 1 of the staged plan (D-024) = RQ1 with patch-level states. It refines `docs/research_plan.md` v1.0 §1 (H1a–H1c), §2 and §5 without changing them; where this document is silent, the research plan applies.

## 1. Arms
| Id | Model | Selection |
|---|---|---|
| PathJev-P | MedGemma-1.5-4B + head + LoRA, OmniMed-Jev recipe, patch states | checkpoint by D-020 rule (min mean val NLL over Choice decisions) |
| (a) | released OmniMed-Jev-4B, zero-shot, 448 px | none |
| (b) | Gen-SFT, same mix, same optimiser steps (D-020) | same D-020 rule, candidate log-likelihood read-out |
| (c) | CONCH v1 zero-shot, prompt ensemble | none |
| (d) | logistic probe on H-optimus-0 features (reference, not a competitor) | C and temperature by grouped CV on training data; cross-fitting where no training labels exist |
| (e) | Inner-Jev letter read-out of the base model (D-019) | none |

All arms are scored on the same state files (`pathjev.evalsets`), with the same candidate sets and texts.

## 2. Test sets and their role
| Test set | Role in Stage 1 | Why |
|---|---|---|
| HepatoBench (liver) | **clean, unseen domain** | organ, classes and source not in training; not in backbone/OmniMed-Jev documented data |
| HiESD (stomach) | **clean, unseen domain** | same |
| BRACS test | **clean, in-domain** | patient-disjoint from BRACS train, which is in the training mix |
| SICAPv2 test | **clean, in-domain** | patient-disjoint from SICAPv2 train, which is in the training mix |
| SPIDER ×4 test | seen-source (secondary) | slide-disjoint re-split (D-018) but same source; SPIDER ⊂ HISTAI |
| later (Phase 4, if obtained): PAR, HunCRC, TissueNet, IMP-CRS2024, GrandQC test | clean, unseen domain | added to the unseen-domain group with no other change |

Pilot (Phase 3): class-stratified subsets (`configs/eval/pilot-stage1.yaml`, ≤ 2,000 states per clean set). Phase 4: full test files.

## 3. Endpoints
Unit of analysis: one (test set, question). For each test set the **main question** is its tissue/lesion Choice (`lesion` BRACS, `gleason_pattern` SICAPv2, `tissue` HepatoBench/HiESD/SPIDER).

- **Primary (unchanged from research_plan §1):** ECE (15 equal-width bins, top-label) and balanced accuracy of the main Choice question, macro-averaged over the clean test sets (unseen-domain + in-domain).
- **Pre-specified secondary breakdowns:** the same two metrics macro-averaged (i) over unseen-domain clean sets only and (ii) over in-domain clean sets only. Stage 1's added value is expected to show in (i); this breakdown is reported whatever its direction.
- **Secondary metrics:** accuracy, macro-F1, adaptive ECE, Brier, NLL, AURC, reliability diagrams; per-set values for every arm.
- **Other questions (secondary, descriptive):** `organ` (single gold class per set; accuracy and ECE only; for HepatoBench/HiESD the gold organ never appears as a training answer, so this measures open-candidate generalisation); Noul `present:*` (AUROC, ECE); Score questions (expected-level MAE, RPS).
- **Missing predictions:** a state without a distribution for a question is counted as wrong and given a uniform distribution; the missing count is reported. A run with > 1 % missing on any set is invalid and is re-run, not analysed.

## 4. Hypotheses and tests (Phase 4, full test sets, 3 seeds)
- **H1a** and **H1b** exactly as in research_plan §1, on the primary endpoint.
- Paired cluster bootstrap over the groups recorded in the split manifests (patient, else slide, else sample), 10,000 resamples, resampling within each test set and recomputing the macro average.
- Seeds: predictions are not averaged across seeds. Each seed is analysed separately; a hypothesis is supported only if it holds for the seed-mean metric *and* in at least 2 of 3 seeds. Mean ± SD across seeds is reported.
- Holm correction over the 4 comparisons of H1a and the 4 of H1b separately.
- **H1c**: gap to (d) with CI, reported; no claim.

## 5. Stage 1 GATE rule (D-024), decided before results
Using the Phase 4 Stage 1 results:
- **Go to Stage 2** if H1a is supported against (c) and (b) *or* PathJev-P beats (c) in macro balanced accuracy (CI > 0), **and** on the unseen-domain breakdown PathJev-P's ECE is not worse than the best zero-shot arm by more than 0.02.
- **Incremental only** (report to the user; venue likely workshop/journal) if the gains exist only in the in-domain breakdown.
- **Stop / redirect** (idea.md falsification) if PathJev-P neither beats (c) in accuracy nor shows a calibration advantage over (a), (b), (c), (e).
The user decides at the GATE; this rule is the recommendation the agent will apply, not an automatic switch.

## 6. Pilot (Phase 3) — what it can and cannot show
1 seed, 1 epoch, subsets: **no hypothesis is tested on the pilot** and no pilot number goes in the paper. The pilot checks:
1. every job completes; wall time, GPU-h and memory per arm (budget update);
2. PathJev-P validation NLL decreases during Stage B;
3. sanity: PathJev-P balanced accuracy on the in-domain sets (BRACS, SICAPv2) is above chance and above (a). If it is not, the agent treats it as a probable bug and investigates before anything else;
4. (a) is reported with and without the appended "other" option and with permuted candidate order (HANDOVER: it picks "other" very often).
Pilot results may change engineering settings (batch size, save interval, time limits). They may not change test sets, endpoints, hypotheses or the GATE rule without a dated amendment approved by the user.

## 7. Calibration-by-temperature analyses (secondary)
- **Approved (research_plan §2):** post-hoc temperature scaling fitted on the validation mix, applied to every arm, reported next to the raw numbers.
- **Proposed in D-024, not yet approved:** temperature fitted on one domain and applied to another (e.g. fitted on BRACS/SICAPv2 validation, tested on HepatoBench/HiESD), to test whether calibration transfers across domains. Run only if the user approves it; if run, reported regardless of outcome.

## 8. What will not be done
No test-set-dependent choice of checkpoint, temperature, prompt, candidate wording or seed; no dropping of test sets or questions after seeing results; no re-definition of "clean" after the fact. Test manifests are hashed (`data/manifests/EXPECTED.sha256`, evaluation-set `info.json`) and every run logs them.

## Amendments
**A1 — 2026-10-20, before any Phase 4 run (approved by the user; D-028).** Reason: compute. The Gen-SFT candidate-likelihood read-out costs ≈ 5.9 s/state on H100 (one full forward per candidate; the cached-prefix variant failed its pre-set equivalence criterion, D-027/D-028), so full test files (≈ 152k states × 3 seeds) would take ≈ 750 GPU-h for that arm alone. Change to §2/§3:
- **Primary analysis sets (all arms, identical states):** BRACS test and SICAPv2 test in full; HepatoBench, HiESD and each SPIDER test capped at **5,000 states** by the class-stratified draw of `pathjev.evalsets` (seed fixed in `configs/eval/stage1-test.yaml`). All endpoints, hypotheses, tests and the GATE rule (§3–§5) are computed on these sets.
- **Secondary:** the same metrics on the full test files for every arm except Gen-SFT (`configs/eval/stage1-test-full.yaml`), reported as a secondary table.
- Nothing else changes. Checkpoint selection evaluates one checkpoint per epoch (D-027 B); the D-020 rule is unchanged.

**A2 — 2026-10-21, POST-HOC: written after the interim seed-1 results were seen (approved by the user; D-036).** Reason: the interim §7 analysis showed that a temperature fitted on the validation mix (whose domains are all seen in training) leaves PathJev-P over-confident on the unseen-domain sets, and that temperature-scaled zero-shot arms then reach lower NLL there. This adds the §7 analysis that D-024 had proposed, fixed here before it is run; it is reported as post-hoc, for every arm, whatever its outcome, and it changes nothing in §3–§5.
- **T-in:** one temperature per arm and seed fitted on the validation Choice decisions of BRACS and SICAPv2 only (in-domain clean sources), applied to the unseen-domain sets.
- **T-cross (leave-one-domain-out):** temperature fitted on the main question of one unseen-domain test set and applied to the other (HepatoBench → HiESD and HiESD → HepatoBench); the unseen macro averages the two transferred results. This uses labels of the *other* test set and is an analysis of transfer, not a deployable procedure.
- **T-oracle:** temperature fitted and evaluated on the same set (upper bound of what one temperature can do; not deployable).
- Metrics: ECE, NLL, Brier, balanced accuracy (unchanged by T), per set and unseen macro, per seed and seed mean. Descriptive only: no hypothesis test, no effect on the GATE rule. Same grid as §7 (`analysis/stage1_analysis.py`, T ∈ [0.05, 200]).

**A3 — 2026-10-22, POST-HOC: written after the full Stage 1 results and an external review were seen (approved by the user; D-040).** Reason: the review questioned the independence of HepatoBench images, sensitivity to candidate order, the generative read-out and the OmniMed-Jev input size. Every analysis below is fixed here before it is run, is reported whatever its outcome, is labelled post-hoc and descriptive (except A3.1(a), a sensitivity re-test of H1a/H1b), and changes nothing in §3–§5 or in the primary results. GPU budget for A3: ≈ 70 GPU-h; if the projection exceeds 84 GPU-h the agent stops and asks.
- **A3.1 HepatoBench grouping.** Images are grouped in blocks of 50 consecutive indices within each class (file stem `NN_CLS_<i>` → block ⌊i/50⌋); sensitivity with blocks of 200. (a) H1a/H1b re-tested exactly as in §4 with these groups (10,000 resamples). (b) All clean-set endpoints and comparisons recomputed without HepatoBench (unseen = HiESD only). (c) The probe re-cross-fitted on HepatoBench with block groups (same nested procedure, `pathjev.baselines`).
- **A3.2 Blank-image control.** PathJev-P (seed 1), Gen-SFT (seed 1) and OmniMed-Jev on 100 states per clean test set with every image replaced by a constant mid-grey image. Expected: identical output within each set; reported: maximum total-variation distance between outputs within a set, and balanced accuracy.
- **A3.3 OmniMed-Jev at 896.** OmniMed-Jev with images pre-resized to 896 instead of 448, on the capped test sets; same metrics as Table main.
- **A3.4 Gen-SFT read-out variants.** From the same forward passes, seeds 1–3, main Choice of the capped clean sets: (i) the primary score; (ii) (i) + log p(`<end_of_turn>`) after the candidate; (iii) (i) divided by the number of candidate tokens. Balanced accuracy, ECE and NLL per set and macro (clean, unseen, in-domain).
- **A3.5 Candidate order.** Ten fixed random permutations of the Choice candidates (seeds `perm-1` … `perm-10`, same permutation for every method) on the capped clean sets: PathJev-P (3 seeds) and OmniMed-Jev with all ten, Gen-SFT (seed 1) with the first five. Per method: clean-macro balanced accuracy and ECE per permutation (mean, SD, range), and on the main Choice the mean total-variation distance between a state's output under a permutation and under the original order, the argmax flip rate, and the frequency of "other" as the argmax.
