# Reviewer 1 Comment 2 reanalysis

The separate **Reviewer 1 Comment 2** Streamlit page runs a traceable analysis.
The working `app.py` is byte-for-byte unchanged. Streamlit discovers the new page
in `pages/2_Reviewer1_Comment2.py`; its validation engine is a separate module.
Existing application functions and old result calculations are not replaced.

The page can also be run independently from the repository root:
`streamlit run pages/2_Reviewer1_Comment2.py`.

The supplied raw, derivative and ratio representations enter unchanged. Invalid
features and inconsistent animal labels are rejected. There are no additional
PMMA/IQR exclusions, derivative generation or missing-value replacements. Historical
exclusions cannot be reconstructed from the prepared workbook; the exported
Methods and response explicitly retain that limitation.

Sampling uses 250 observations per animal by default, without replacement, seed
42 within each animal, before cross-validation. Every model uses the same selected
rows within a representation. Source Excel rows and the input hash are exported.

Each outer LOGO fold holds out one animal. Scaling is fitted on training animals
only. The five classifiers have fixed configurations, with complete resolved
parameters saved. PLS component count is selected from 1–5 in an inner LOGO loop
restricted to the outer training animals, with scaling refitted for every inner
fold. The highest mean inner-animal accuracy wins; ties within 1e-12 choose the
smallest component count. A separate fixed-three-component result is retained.

The candidate grid is an explicit specification for this new analysis, not an
explanation of the historical choice of three components. Component counts can
vary between outer folds; do not describe the nested analysis as fixed at three.

The spectral ranges remain Fingerprint 800–1800, Collagen 1300–1700 and Mineral
800–1200 cm⁻¹. The existing manuscript range discrepancy and ratio-header
discrepancies require author resolution and are not silently changed by this panel.

Excel contains summary, animal results, component selection, inner scores, input
audit, selected rows and complete run settings. The ZIP adds every OOF prediction,
fitted model configurations, warnings, exact software versions, Methods and a
reviewer-response draft. Current workbook, settings and engine identity must match
the completed run before downloads are offered.

Tests: `python -m unittest discover -s tests -v`.

## Numerical verification on the supplied study workbook

An independent rerun used the supplied workbook with SHA-256
`ea87e1071add346dfa754ec31e9b185b1383dfcfea84b0a110b7248d2b8376c1`.
It produced 49 model/representation summaries and 588 held-out-animal results.
Raw fingerprint SVM, KNN, logistic regression and XGBoost reproduced the archived
overall accuracies and all 12 animal accuracies exactly to tolerance 1e-12.
Random forest gave 87.2000%, versus archived 87.1667%: one additional correct
spectrum out of 3,000. This is a small numerical difference, not exact reproduction.

Across all 35 fixed-classifier comparisons, 27 overall means and 394 of 420
animal accuracies matched. The largest overall difference was 0.20 percentage
points. The six spectral fixed-three PLS accuracies matched the archived PLS
summary. Historical dependency versions and causes of the remaining numerical
differences were not established. The new run records its own exact versions;
do not claim that every archived prediction or the original runtime was recovered.

## Repository audit of upstream preprocessing

The main history at `d57621b4cb40bcf95345d4d82e0b8c0dba6c240e` contains 31
commits, 30 distinct file trees and 13 distinct application-source versions.
The current main history, the two additional branches and historical documentation
were checked. No original exclusion log, laboratory preprocessing project, or
input workbook was found among those versioned files. GitHub Releases is empty.

The IQR calculations at
[`app.py` lines 11598 onward](https://github.com/wavhalkomal/bonespec-ai/blob/d57621b4cb40bcf95345d4d82e0b8c0dba6c240e/app.py#L11598)
are inside `_q2a_figure`. They select graph limits and annotate observations outside
the plotted view. They do not remove observations from the classifier training or
test datasets. No PMMA-specific rejection procedure was identified in the 13
historical application versions. Generic missing-value handling is a separate
operation and is not evidence of PMMA or IQR exclusion.

The application has optional Savitzky–Golay helpers in exploratory analysis
sections, with defaults of window length 7, polynomial order 3, derivative order 2,
and `mode="interp"`. Their existence does not establish that they generated the
uploaded second-derivative worksheet. The grouped classifier path reads that
worksheet directly and does not call those helpers.

An [earlier README](https://github.com/wavhalkomal/bonespec-ai/blob/e52c944ebcea673642418a1bcb400c118b5b3a46/README.md#L101)
also distinguishes manuscript-described laboratory preprocessing from the
operations reproduced by the application.

The [originating study](https://pmc.ncbi.nlm.nih.gov/articles/PMC11539169/) supplies
part of the protocol: PMMA screening at intensity(1735) > 0.5 × intensity(1660),
Savitzky–Golay smoothing and second derivatives, and 1.5×IQR fences for peak-ratio
datasets. This published protocol is not a per-observation processing log for the
current workbook. The IQR grouping, exact derivative settings and effects on the
classification rows remain unverified. Only describe these historical steps as
verified for the workbook when the preparation record supports that statement.

The corrected [reviewer response](REVIEWER1_COMMENT2_RESPONSE.md) reports the known
source protocol and the verified evaluation separately. No additional exclusion or
derivative step was added, and no original working-script code was modified.
