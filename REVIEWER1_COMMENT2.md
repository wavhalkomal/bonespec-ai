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
