# Release history

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
