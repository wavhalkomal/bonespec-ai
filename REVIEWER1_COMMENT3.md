# Reviewer 1 Comment 3

This adds a separate Streamlit page for animal-level zone-Shapley stability and
reference sensitivity. The working `app.py`, existing classifiers and requirements
are unchanged. The page can also run independently:

`streamlit run pages/3_Reviewer1_Comment3.py`

Upload Raw and/or 2nd derivative sheets with numeric wavenumber columns, animal IDs
and WT/OIM phenotype. The default new sensitivity analysis uses the raw full
fingerprint, LR and XGBoost, 250 spectra per animal, seed 42, ten primary explained
spectra and fifty for the explanation-sampling check. No additional contamination
screening, outlier exclusion, smoothing or derivative generation is applied.

The nine original chemical zones plus a gaps group assign every channel exactly
once. Shared endpoints use the original first-match rule. LOGO separates animals;
StandardScaler and every reference are fitted on training animals only. The model
and explained spectra are fixed across reference comparisons. All 2^K coalitions
are enumerated on the P(OIM) scale.

Reference alternatives are the training mean, coordinate-wise training median,
and three empirical panels. Each panel contains one full spectrum from each
training animal. Coalition probabilities are averaged over donors; evaluating an
averaged spectrum would be a different calculation. The primary ten spectra are
contained in the fifty-spectrum mean-reference check. Training and explanation
sampling reproduce the repository's deterministic zone-SHAP policy; exact original
historical spectrum IDs and runtime versions were not recovered.

Downloads contain CSV tables, every per-spectrum contribution, actual reference
outputs, selected source rows, donor rows, fitted settings, versions and Methods.
Results disappear when input or configuration no longer matches the completed run.
No fixed 0.5 baseline is used to reconstruct predictions. The existing main app's
waterfall plotting branch does use 0.5; that presentation does not affect its
stored zone contributions and must not be used as evidence for a fold's actual
reference output.

## Results on the supplied study records

The archived Method B raw outputs put Amide I, Amide II and PO4 in the top three
for 11/12, 10/12 and 10/12 animals with LR and 12/12, 11/12 and 10/12 with XGBoost.
Median pairwise animal-rank correlations are 0.855 and 0.885. Amide I falls to rank
eight for A103 under LR, so the pooled ranking is not universal.

The new matched sensitivity analysis retains those three globally leading regions
for both models with every tested background. Median within-animal rank correlations
for mean versus median are 0.988 (LR) and 0.976 (XGBoost). Across three empirical
panels, median correlations range from 0.945–0.976 and 0.939–0.964, respectively.
Magnitudes and some animal-level top-three sets change; no universal background
invariance is claimed. This is a new analysis, separate from the archived run.

Fold and animal variation are paired in LOGO, and the training sets overlap.
Rank summaries are descriptive; the biological sample size remains twelve.
Mean/median references and hybrid spectra are not guaranteed to be physically
realizable. Grouping preserves the reference pattern within a replaced zone but
can break cross-zone dependencies. This is not conditional or causal SHAP.

## Reproduction and checks

`python -m unittest discover -s tests -v` runs 17 checks, including independent
permutation-based Shapley verification, training/held-out separation, background
averaging, matched explained samples, unchanged predictions and UI result identity.

To summarize an archived Method B workbook without refitting:

`python research/analyze_archived_zone_shap.py input.xlsx --representation Raw --output archived_summary`

The reader keeps Method A separate and verifies rank concordance independently.
No private input data or archived workbooks are committed. The supplied input
SHA-256 is `ea87e1071add346dfa754ec31e9b185b1383dfcfea84b0a110b7248d2b8376c1`.

The [reviewer response](REVIEWER1_COMMENT3_RESPONSE.md) gives the measured results
and limits. Methodological reference: Contreras et al., Analytical Chemistry 96
(2024), 15588–15597, https://doi.org/10.1021/acs.analchem.4c02329.
