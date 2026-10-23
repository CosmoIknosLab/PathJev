# data/manifests

- `<dataset>.csv`: patient- or slide-level split manifests. Hashes are in `EXPECTED.sha256`.
- `hiesd_patient_slide_mapping.csv`: the official HiESD slide→patient map (104 slides, 44 patients). It comes from https://huggingface.co/datasets/JSGe-AI/HiESD, file `patient_slide_mapping.csv`, revision `89878d1841ef7ca3830ba23f7844bac4cc0cdb7a`. Its SHA-256 is `f3ab7a37c7915d5c241a3e1399c8b428c03061cd6086f364b0f29aa3e983db67`. Downloaded 2026-10-23 and stored unchanged (UTF-8 with BOM). The Stage 1 analysis uses patient_id as the bootstrap and probe cross-fit group for HiESD (plan §3: patient, else slide, else sample).
- `stage1_test_exclusions.csv`: test states removed for every method because they duplicate a training or validation image (near-duplicate audit, job 7548; `results/runs/stage1-dedup/`). `scope` says whether the capped primary sets, the full secondary files, or both are affected.
