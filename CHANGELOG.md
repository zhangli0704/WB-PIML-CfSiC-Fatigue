# Release history

## 1009 environment-shared residual release

- Synchronize the 224-record dataset, latest analysis and all 290 artifacts from
  `full_20261008_182317_470433` with the October 9 IJF revision.
- Restore the Min 2013 runout; retain 158 fractures and 66 right-censored records.
- Apply the supplied environment-shared residual rules throughout selection,
  calibration, primary analysis, sensitivity, strict holdouts and frozen inference.
- Divide the working script into twelve short files with minimal comments,
  preserving algorithm syntax, source order, equations, candidates and seeds.
- Record original/public source hashes; adapt source-integrity bookkeeping and
  usage instructions to the public entry point.
- Preserve original artifacts, with lossless gzip for strict calibration; update
  README scores, dependency pins and verification.
- Record a full rerun of the public source and numeric comparisons with the
  supplied primary, sensitivity, strict-holdout and prospective results.
- Correct the launcher so prospective prediction/evaluation does not receive
  the primary-analysis-only default data argument.



## 0930 revision release

- Synchronize the exact 8,835-line source and complete 289-file run from
  `0928_shuiyangxin/runs/full_20260928_160433_681221`.
- Replace four old fragments with three byte-preserving source fragments. Use a
  hash-verified runtime reconstruction to preserve source self-checks and workers.
- Include learned-water training, 34 conditional sensitivity variants, strict
  campaign-separated factor holdouts, censoring evidence audits and frozen-model
  prediction/evaluation. These are the supplied revision, not packaging changes
  to the mathematical model.
- Move primary artifacts to `reference_results/primary/` and include complete
  `reviewer/` and `prospective/` directories, original logs and provenance.
- Update the README to the September 30 manuscript, actual executed environment,
  revised metrics, figure numbering correspondence and disclosed limitations.
- Verify saved metrics, all sensitivity checkpoints and strict partitions;
  additionally rerun the five selected pipelines and their inner calibration fits,
  plus the selected frozen final fit. Do not claim a new complete nested search.
- Preserve the 87 MB strict-calibration CSV through lossless gzip; automatically
  verify its original decompressed hash and expand it for temporary audits.
  Keep the other 288 original run files unchanged and disable Git line-ending
  transformations so published SHA-256 hashes also hold after checkout.

## 0912 repository release

- Packaged the revised-manuscript release in a new local checkout while retaining
  the supplied 0909 canonical dataset and numerical artifacts, which already match
  the current remote repository byte-for-byte.
- Split the 6368-line canonical analysis into four reviewable source fragments.
  Their ordered byte concatenation exactly reconstructs the supplied source, so the
  executed statements, constants, functions and entry point are unchanged.
- Replaced the long top-level source with a small compatibility loader that verifies
  fragment hashes before executing them in one shared namespace.
- Extended release verification to check the fragment reconstruction hash before
  loading the analysis and recomputing saved-result consistency checks.

## Changes from the 0903 repository to the 0909 release

- Replaced the previous 222-record dataset (160 exact, 62 runout) with the supplied
  223-record dataset (158 exact, 65 runout), including its current provenance sheets.
- Replaced the old `data.py` / `models.py` / `validation.py` / `run.py` implementation
  with the exact 0909 `WB-PIML.py` source and a small compatibility `run.py` launcher.
  This is a source synchronization, not a new model fitted during packaging.
- Synchronized current model selection, probability evaluation and simple-trunk
  controls with the supplied 0909 code and protocol.
- Replaced 45 old result worksheets with the 57-sheet reference workbook; added
  all ten PNG outputs and the supplied 29-slide presentation.
- Updated dependencies to include Matplotlib, Pillow and python-pptx, which are
  required by the current figure/presentation output path.
- Added integrity and saved-result consistency checks, hashes, method scope and
  limitations. Corrected only the reference README's inherited figure-count sentence
  and added a packaging note. Source code and numerical artifacts remain unchanged.

The previous release remains available in repository history, including commit
`39f3a93a6ba3998cafb98276855b2bd455a86b76` before this update.
