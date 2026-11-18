"""Audited reanalysis for Reviewer 1, Comment 2.

Prepared representations enter unchanged. Scaling and PLS component selection
are fitted on training animals only. Historical exclusion provenance is never
inferred from the prepared workbook. This module does not alter legacy results.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import time
import warnings
import zipfile

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits


ENGINE_VERSION = "r1c2-1.0.1"
BASE_APP_BLOB = "0e67a5bf9ab52c98c0f2296544da30a8595827bd"
MODEL_NAMES = ("SVM", "KNN", "Logistic Regression", "Random Forest", "XGBoost")
SHEET_NAMES = ("Raw", "2nd derivative", "Quant peak ratios")
REGIONS = (("Fingerprint", 800, 1800), ("Collagen", 1300, 1700), ("Mineral", 800, 1200))


@dataclass(frozen=True)
class AnalysisConfig:
    spectra_per_animal: int = 250
    seed: int = 42
    candidates: tuple[int, ...] = (1, 2, 3, 4, 5)
    models: tuple[str, ...] = MODEL_NAMES
    regions: tuple[tuple[str, int, int], ...] = REGIONS
    upstream_notes: str = ""

    def validate(self):
        if not isinstance(self.spectra_per_animal, int) or self.spectra_per_animal < 1:
            raise ValueError("Spectra per animal must be a positive integer.")
        if not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError("Seed must be an integer between 0 and 2**32-1.")
        if not self.candidates or any(not isinstance(k, int) or k < 1 for k in self.candidates):
            raise ValueError("PLS candidates must be positive integers.")
        if tuple(sorted(set(self.candidates))) != self.candidates:
            raise ValueError("PLS candidates must be unique and increasing.")
        if len(set(self.models)) != len(self.models) or set(self.models) - set(MODEL_NAMES):
            raise ValueError("Unknown or duplicate classifier selection.")
        if not self.regions or len({r[0] for r in self.regions}) != len(self.regions):
            raise ValueError("Feature-region names must be unique.")
        for name, lo, hi in self.regions:
            if not name or not np.isfinite([lo, hi]).all() or lo >= hi:
                raise ValueError("Each region needs a name and increasing finite bounds.")


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def engine_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def run_fingerprint(data: bytes, config: AnalysisConfig):
    payload = {"input_sha256": hashlib.sha256(data).hexdigest(),
               "configuration": asdict(config), "engine_sha256": engine_hash()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _metadata_columns(frame):
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate column names are not supported.")
    norm = {str(c).strip().lower(): c for c in frame.columns}
    group = next((norm[k] for k in ("sample id", "animal id", "mouse id", "id") if k in norm), None)
    label = next((norm[k] for k in ("type", "class", "label", "group", "genotype") if k in norm and norm[k] != group), None)
    if group is None or label is None:
        raise ValueError("Each sheet needs an animal-ID column and a phenotype column, such as Sample ID and Type.")
    return group, label


def _wavenumber(column):
    try:
        return float(str(column).strip()) if "/" not in str(column) else None
    except ValueError:
        return None


def prepare_views(frames: dict[str, pd.DataFrame], config: AnalysisConfig):
    """Strict deterministic sampling with source-row traceability and no QC exclusions."""
    config.validate()
    views, audit, selected = [], [], []
    reference_cohort = None
    if not frames or set(frames) - set(SHEET_NAMES):
        raise ValueError("Supply one or more of Raw, 2nd derivative and Quant peak ratios.")
    for sheet in SHEET_NAMES:
        if sheet not in frames:
            continue
        frame = frames[sheet].copy()
        group_col, label_col = _metadata_columns(frame)
        if frame.empty:
            raise ValueError(f"{sheet}: no observations.")
        if frame[[group_col, label_col]].isna().any().any():
            raise ValueError(f"{sheet}: missing animal ID or phenotype; no rows were silently removed.")
        frame[group_col] = frame[group_col].astype(str).str.strip()
        frame[label_col] = frame[label_col].astype(str).str.strip().str.upper()
        if (frame[group_col] == "").any() or (frame[label_col] == "").any():
            raise ValueError(f"{sheet}: blank animal ID or phenotype.")
        if set(frame[label_col]) != {"WT", "OIM"}:
            raise ValueError(f"{sheet}: this study workflow requires exactly WT and OIM phenotypes.")
        group_labels = frame.groupby(group_col)[label_col].agg(lambda s: sorted(set(s)))
        bad = group_labels[group_labels.map(len) != 1]
        if len(bad):
            raise ValueError(f"{sheet}: conflicting phenotypes within animal(s): {list(bad.index)}.")
        cohort = {str(a): labels[0] for a, labels in group_labels.items()}
        if reference_cohort is not None and cohort != reference_cohort:
            raise ValueError(f"{sheet}: animal IDs or phenotypes differ between supplied representations.")
        reference_cohort = cohort
        if min(pd.Series(cohort).value_counts()) < 3:
            raise ValueError("Nested animal-grouped selection requires at least three animals per phenotype.")
        counts = frame.groupby(group_col).size()
        short = counts[counts < config.spectra_per_animal]
        if len(short):
            raise ValueError(f"{sheet}: fewer than {config.spectra_per_animal} rows for {short.to_dict()}; no unequal fallback was used.")
        metadata = {group_col, label_col}
        if sheet == "Quant peak ratios":
            feature_columns = [c for c in frame if c not in metadata and "/" in str(c)]
            regions = [("Peak ratios", feature_columns)]
        else:
            spectral = [(c, _wavenumber(c)) for c in frame if c not in metadata]
            spectral = [(c, w) for c, w in spectral if w is not None]
            if len({w for _, w in spectral}) != len(spectral):
                raise ValueError(f"{sheet}: duplicate numeric wavenumbers.")
            regions = [(name, [c for c, w in spectral if lo <= w <= hi]) for name, lo, hi in config.regions]
            feature_columns = list(dict.fromkeys(c for _, cols in regions for c in cols))
        if not feature_columns or any(len(cols) < 3 for _, cols in regions):
            raise ValueError(f"{sheet}: each analyzed feature set requires at least three numeric features.")
        numeric = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
        valid = np.isfinite(numeric.to_numpy(dtype=float))
        if not valid.all():
            row, col = np.argwhere(~valid)[0]
            raise ValueError(f"{sheet}: invalid/missing feature at Excel row {int(row)+2}, column {feature_columns[int(col)]!r}; no zero replacement was performed.")
        frame[feature_columns] = numeric
        # Index positions are the only source-row IDs available in this workbook.
        frame = frame.reset_index(drop=True)
        balanced = pd.concat([g.sample(n=config.spectra_per_animal, random_state=config.seed)
                              for _, g in frame.groupby(group_col, sort=True)])
        source_rows = balanced.index.to_numpy(dtype=int) + 2
        labels = balanced[label_col].to_numpy()
        groups = balanced[group_col].to_numpy()
        selected.append(pd.DataFrame({"Sheet": sheet, "Source Excel row": source_rows,
                                      "Animal": groups, "Phenotype": labels}))
        for animal, count in counts.items():
            audit.append({"Sheet": sheet, "Animal": animal, "Phenotype": cohort[animal],
                          "Rows available as supplied": int(count), "Additional QC exclusions": 0,
                          "Rows selected": config.spectra_per_animal,
                          "Sampling stage": "Within animal, before outer LOGO",
                          "Upstream PMMA/IQR history": "Not independently reconstructed"})
        for region, columns in regions:
            views.append({"name": f"{sheet} | {region}", "sheet": sheet, "region": region,
                          "features": [str(c) for c in columns],
                          "X": balanced[columns].to_numpy(dtype=float),
                          # Preserve the original grouped workflow's alphabetical encoding.
                          "y": (labels == "WT").astype(int), "groups": groups.copy(),
                          "source_rows": source_rows.copy()})
    return views, pd.DataFrame(audit), pd.concat(selected, ignore_index=True)


def read_workbook(data: bytes, config: AnalysisConfig):
    with pd.ExcelFile(io.BytesIO(data)) as book:
        names = {str(n).strip().casefold(): n for n in book.sheet_names}
        if len(names) != len(book.sheet_names):
            raise ValueError("Duplicate sheet names after case/whitespace normalization.")
        frames = {canonical: book.parse(names[canonical.casefold()]) for canonical in SHEET_NAMES
                  if canonical.casefold() in names}
    return prepare_views(frames, config)


def build_models(config):
    models = {"SVM": SVC(cache_size=1000),
              "KNN": KNeighborsClassifier(n_neighbors=3, n_jobs=1),
              "Logistic Regression": LogisticRegression(max_iter=1000, random_state=config.seed),
              "Random Forest": RandomForestClassifier(n_estimators=100, random_state=config.seed, n_jobs=1)}
    if "XGBoost" in config.models:
        try:
            from xgboost import XGBClassifier
        except ImportError as error:
            raise ValueError("XGBoost was selected but is unavailable. Install the repository requirements or explicitly deselect it.") from error
        models["XGBoost"] = XGBClassifier(eval_metric="logloss", random_state=config.seed, n_jobs=1)
    return {name: models[name] for name in config.models}


def _fit(model, X, y, context, warning_records):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X, y)
    for warning in caught:
        record = {**context, "Category": warning.category.__name__, "Message": str(warning.message)}
        warning_records.append(record)
        if issubclass(warning.category, ConvergenceWarning):
            raise ValueError(f"A model did not converge ({context}); this run was not accepted.")
    return model


def select_pls_components(X, y, groups, candidates, outer_animal, warning_records=None):
    """Receives only outer-training arrays; every inner scaler is refitted."""
    warning_records = [] if warning_records is None else warning_records
    splits = list(LeaveOneGroupOut().split(X, y, groups))
    limit = min(X.shape[1], min(len(tr)-1 for tr, _ in splits))
    eligible = tuple(k for k in candidates if k <= limit)
    if not eligible:
        raise ValueError("No candidate PLS component count is feasible in every inner training fold.")
    records = []
    for tr, va in splits:
        if len(np.unique(y[tr])) != 2:
            raise ValueError("An inner training fold contains only one phenotype.")
        scaler = StandardScaler().fit(X[tr])
        Xtr, Xva = scaler.transform(X[tr]), scaler.transform(X[va])
        inner_animal = str(groups[va][0])
        training_animals = sorted(map(str, np.unique(groups[tr])))
        for k in eligible:
            model = _fit(PLSRegression(n_components=k, scale=False), Xtr, y[tr],
                         {"Stage": "Inner PLS", "Outer animal": outer_animal,
                          "Inner animal": inner_animal, "Components": k}, warning_records)
            pred = (np.asarray(model.predict(Xva)).ravel() >= .5).astype(int)
            records.append({"Outer held-out animal": outer_animal, "Inner held-out animal": inner_animal,
                            "Components": k, "Inner accuracy": float(accuracy_score(y[va], pred)),
                            "Inner test rows": len(va), "Inner training animals": json.dumps(training_animals)})
    scores = pd.DataFrame(records)
    means = scores.groupby("Components")["Inner accuracy"].mean()
    best = float(means.max())
    chosen = min(int(k) for k, score in means.items() if np.isclose(score, best, rtol=0, atol=1e-12))
    choices = [{"Outer held-out animal": outer_animal, "Components": int(k),
                "Mean inner animal accuracy": float(score), "Selected": int(k) == chosen,
                "Inner animal folds": len(splits)} for k, score in means.items()]
    return chosen, scores, pd.DataFrame(choices)


def analyze_views(views, input_audit, selected_rows, config, input_sha256, input_name="uploaded.xlsx", progress=None):
    """New fixed-classifier and nested-PLS run; never overwrites legacy results."""
    config.validate()
    model_prototypes = build_models(config)
    fold_rows, predictions, inner_rows, choice_rows, fit_configs, warning_records = [], [], [], [], [], []
    total = sum(len(np.unique(v["groups"])) for v in views)
    completed = 0
    for view in views:
        X, y, groups = view["X"], view["y"], view["groups"]
        for tr, te in LeaveOneGroupOut().split(X, y, groups):
            animal = str(groups[te][0])
            warning_start = len(warning_records)
            chosen, inner, choices = select_pls_components(X[tr], y[tr], groups[tr], config.candidates, animal, warning_records)
            for warning in warning_records[warning_start:]:
                warning.setdefault("Feature set", view["name"])
            inner.insert(0, "Feature set", view["name"]); choices.insert(0, "Feature set", view["name"])
            inner_rows.append(inner); choice_rows.append(choices)
            scaler = StandardScaler().fit(X[tr])
            Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])
            models = {name: clone(model) for name, model in model_prototypes.items()}
            models["PLS-DA (fixed 3)"] = PLSRegression(n_components=3, scale=False)
            models["PLS-DA (nested selection)"] = PLSRegression(n_components=chosen, scale=False)
            for name, model in models.items():
                context = {"Stage": "Outer model", "Feature set": view["name"], "Animal": animal, "Model": name}
                _fit(model, Xtr, y[tr], context, warning_records)
                raw = np.asarray(model.predict(Xte)).ravel()
                pred = (raw >= .5).astype(int) if isinstance(model, PLSRegression) else raw.astype(int)
                components = int(model.n_components) if isinstance(model, PLSRegression) else None
                correct = int((pred == y[te]).sum())
                training_animals = sorted(map(str, np.unique(groups[tr])))
                fold_rows.append({"Feature set": view["name"], "Sheet": view["sheet"], "Region": view["region"],
                                  "Model": name, "Held-out animal": animal,
                                  "Phenotype": "OIM" if y[te][0] == 0 else "WT",
                                  "Training animals": json.dumps(training_animals), "Training rows": len(tr),
                                  "Test rows": len(te), "Correct": correct, "Accuracy": correct/len(te),
                                  "PLS components": components})
                predictions.append(pd.DataFrame({"Feature set": view["name"], "Model": name,
                                                  "Source Excel row": view["source_rows"][te],
                                                  "Animal": groups[te], "True code": y[te], "Predicted code": pred,
                                                  "True phenotype": np.where(y[te] == 0, "OIM", "WT"),
                                                  "Predicted phenotype": np.where(pred == 0, "OIM", "WT"),
                                                  "Correct": pred == y[te], "PLS components": components}))
                fitted = {**context, "Parameters": _jsonable(model.get_params(deep=False)),
                          "Training animals": training_animals, "Scaler training rows": len(tr),
                          "Scaler mean sha256": hashlib.sha256(scaler.mean_.tobytes()).hexdigest()}
                if name == "XGBoost":
                    fitted["Resolved booster configuration"] = json.loads(model.get_booster().save_config())
                fit_configs.append(fitted)
            completed += 1
            if progress:
                progress(completed/total, f"{view['name']} — held-out {animal}; selected PLS components={chosen}")
    folds = pd.DataFrame(fold_rows)
    oof = pd.concat(predictions, ignore_index=True)
    summary = []
    for (feature_set, model), f in folds.groupby(["Feature set", "Model"], sort=False):
        p = oof[(oof["Feature set"] == feature_set) & (oof["Model"] == model)]
        if p.duplicated(["Animal", "Source Excel row"]).any() or len(p) != int(f["Test rows"].sum()):
            raise AssertionError("Out-of-fold prediction coverage is inconsistent.")
        accuracy = float(p["Correct"].mean())
        if not np.isclose(accuracy, f["Correct"].sum()/f["Test rows"].sum()):
            raise AssertionError("Fold and pooled accuracy disagree.")
        precision, recall, f1, _ = precision_recall_fscore_support(p["True code"], p["Predicted code"], average="binary", pos_label=0, zero_division=0)
        summary.append({"Feature set": feature_set, "Model": model, "OOF accuracy": accuracy,
                        "Animal mean accuracy": float(f["Accuracy"].mean()), "Animal SD": float(f["Accuracy"].std(ddof=1)),
                        "Animal minimum": float(f["Accuracy"].min()), "Animal maximum": float(f["Accuracy"].max()),
                        "Precision (OIM)": precision, "Recall (OIM)": recall, "F1 (OIM)": f1,
                        "Animal folds": len(f), "OOF observations": len(p)})
    versions = {}
    for package in ("numpy", "pandas", "scipy", "scikit-learn", "xgboost", "openpyxl", "streamlit", "threadpoolctl"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    manifest = {"engine_version": ENGINE_VERSION, "engine_sha256": engine_hash(),
                "original_app_git_blob": BASE_APP_BLOB,
                "input_sha256": input_sha256, "input_name": input_name,
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "python": platform.python_version(), "platform": platform.platform(), "software_versions": versions,
                "configuration": _jsonable(asdict(config)), "positive_class": "OIM", "positive_code": 0,
                "outer_validation": "LeaveOneGroupOut by animal", "inner_validation": "LeaveOneGroupOut within outer training animals",
                "component_selection": "Highest mean inner-animal accuracy; smallest count for ties within absolute tolerance 1e-12",
                "scaler_parameters": StandardScaler().get_params(deep=False), "scaler_variance_ddof": 0,
                "sampling": f"{config.spectra_per_animal} without replacement per animal, before outer CV; pandas random_state={config.seed} for each animal, same selected rows for all models within representation",
                "additional_exclusions": 0, "additional_derivative_or_smoothing": False,
                "historical_PMMA_IQR_reconstructed": False,
                "upstream_description_status": "Author-supplied description, not independently verified" if config.upstream_notes.strip() else "Unavailable",
                "upstream_description": config.upstream_notes.strip(),
                "feature_sets": [{k: v[k] for k in ("name", "sheet", "region", "features")} for v in views],
                "fixed_classifier_parameters": {name: _jsonable(model.get_params(deep=False)) for name, model in model_prototypes.items()},
                "new_analysis": True}
    return {"summary": pd.DataFrame(summary), "folds": folds, "oof": oof,
            "inner_scores": pd.concat(inner_rows, ignore_index=True),
            "component_selection": pd.concat(choice_rows, ignore_index=True),
            "input_audit": input_audit, "selected_rows": selected_rows,
            "fitted_configurations": fit_configs, "warnings": warning_records, "manifest": manifest}


def run_analysis(data: bytes, config=AnalysisConfig(), input_name="uploaded.xlsx", progress=None):
    started = datetime.now(timezone.utc)
    start_clock = time.perf_counter()
    views, audit, selected = read_workbook(data, config)
    # Bound BLAS parallelism; classifier algorithms and their parameters are unchanged.
    with threadpool_limits(limits=2):
        result = analyze_views(views, audit, selected, config, hashlib.sha256(data).hexdigest(), input_name, progress)
    result["manifest"]["run_fingerprint"] = run_fingerprint(data, config)
    result["manifest"]["started_utc"] = started.isoformat()
    result["manifest"]["analysis_elapsed_seconds"] = time.perf_counter() - start_clock
    result["manifest"]["runtime_controls"] = {"BLAS_threads": 2, "classifier_n_jobs": 1}
    return result


def methods_text(result):
    m = result["manifest"]; c = m["configuration"]
    n = c["spectra_per_animal"]; seed = c["seed"]
    candidates = ", ".join(map(str, c["candidates"]))
    animals = int(result["selected_rows"]["Animal"].nunique())
    regions = "; ".join(f"{name}: {lo}–{hi} cm⁻¹" for name, lo, hi in c["regions"])
    return ("# Methods for the new Reviewer 1 Comment 2 analysis\n\n"
            f"The supplied workbook was analyzed as a set of prepared representations. Animal ID was used only for grouping, "
            f"and WT/OIM phenotype was the classification target (OIM=0, WT=1, matching the original grouped implementation). Input validation required finite numeric analyzed features, "
            f"one phenotype per animal, matching cohorts across supplied sheets, and at least {n} observations per animal. "
            "Invalid entries were rejected rather than replaced with zero. No additional PMMA/IQR exclusion, smoothing, "
            "second-derivative generation or imputation was performed in this reanalysis. The original upstream exclusion history "
            "was not independently reconstructed; validation remains conditional on the supplied dataset.\n\n"
            f"Within each representation, {n} observations per animal were sampled without replacement using pandas sampling "
            f"with random_state={seed} for each animal, before cross-validation. All evaluated models used the same selected rows "
            "within that representation. Selected Excel source rows and the input SHA-256 were exported. Sampling across representations "
            f"does not imply matched physical spectra. The spectral ranges were {regions}. Existing peak-ratio headers were retained verbatim.\n\n"
            f"Outer leave-one-group-out cross-validation held out one of the {animals} animals at a time. StandardScaler was fitted "
            "only on outer-training animals and applied to the held-out animal. The classifier configurations were fixed and "
            "were not optimized. All resolved estimator parameters and runtime package versions were exported; trained XGBoost "
            "booster configurations were also saved where XGBoost was evaluated.\n\n"
            f"PLS-DA component count was selected from {candidates} in an inner animal-grouped leave-one-group-out loop "
            "restricted to the outer-training animals. A fresh scaler was fitted within each inner training fold. The component "
            "count maximizing mean inner-animal accuracy was selected, choosing the smallest count in a tie within absolute "
            "tolerance 1e-12. Feasible candidates were bounded by feature and inner-training dimensions. A fresh scaler and "
            "PLSRegression(scale=False) model were then fitted on all outer-training animals. Binary predictions used a fixed "
            "threshold of 0.5. The outer-test animal did not participate in scaling, component selection or fitting. A separate "
            "fixed-three-component PLS-DA evaluation was retained for comparison. The candidate grid was specified for this new "
            "analysis and is not a reconstruction of the historical reason for three components.\n\n"
            "Predictions for all held-out observations were pooled to compute OOF accuracy and phenotype-level precision, recall "
            "and F1. Per-animal accuracies were summarized by their unweighted mean, sample SD and minimum–maximum range. "
            "These results are a new reanalysis and must not be represented as the original submitted run.\n")


def reviewer_text(result):
    models = result["manifest"]["configuration"]["models"]
    classifier_description = ("The evaluated machine-learning classifiers (" + ", ".join(models) + ") use fixed configurations rather than a hyperparameter search. ") if models else "No machine-learning classifiers were selected in this run; it evaluates PLS-DA only. "
    return ("# Response to Reviewer 1 Comment 2\n\n"
            "We thank the reviewer for requesting clarification of preprocessing and model configuration. We performed an "
            "additional animal-grouped validation analysis in which all learned feature scaling and PLS-DA component selection "
            "were restricted to training animals. The supplied representations were sampled within animal before validation. "
            "No additional contamination/outlier exclusion, smoothing or derivative generation was performed in this reanalysis. "
            "We distinguish the prepared input from the operations fitted during validation. The available workbook does not "
            "preserve the complete historical PMMA/IQR exclusion history or derivative settings. Consequently, those earlier "
            "operations cannot be independently audited from this workbook, and we acknowledge that the evaluation is "
            "conditional on the supplied data.\n\n"
            "For the new PLS-DA evaluation, component count was selected by inner leave-one-animal-out cross-validation within "
            "each outer training set. Each inner scaler was fitted exclusively to its training animals. The selected model was "
            "then refitted on the outer-training animals and evaluated on the untouched outer-test animal. The full candidate "
            "grid, selection scores, fold-specific component counts and tie rule are supplied with the results. The earlier "
            "fixed-three-component evaluation was retained as a comparison, and we do not claim that its original count was "
            "historically optimized.\n\n"
            + classifier_description + "The full "
            "estimator settings, software versions, input hash, selected source rows and fitted-model configurations are "
            "recorded with the run. StandardScaler is fitted within each outer fold for all classifiers and PLS-DA. No outer-test "
            "data are used to fit those scaling statistics or to select the PLS component count. The Methods text and the "
            "run-specific results accompanying this response describe this new analysis separately from the original submission.\n")


def result_workbook(result):
    """Application-generated Excel audit report; full OOF data are in the companion ZIP."""
    from openpyxl import Workbook
    from openpyxl.cell import WriteOnlyCell
    from openpyxl.styles import Font, PatternFill, Alignment
    wb = Workbook(write_only=True)
    tables = [("Summary", result["summary"]), ("Animal accuracy", result["folds"]),
              ("PLS selection", result["component_selection"]), ("PLS inner folds", result["inner_scores"]),
              ("Input audit", result["input_audit"]), ("Selected spectra", result["selected_rows"])]
    config_rows = [{"Group": "Manifest", "Parameter": k, "Value": json.dumps(_jsonable(v), sort_keys=True) if isinstance(v, (dict, list)) else v}
                   for k, v in result["manifest"].items()]
    for model, parameters in result["manifest"]["fixed_classifier_parameters"].items():
        config_rows.extend({"Group": model, "Parameter": k, "Value": json.dumps(_jsonable(v))} for k, v in parameters.items())
    tables.append(("Run configuration", pd.DataFrame(config_rows)))
    percent_names = {"OOF accuracy", "Animal mean accuracy", "Animal SD", "Animal minimum", "Animal maximum", "Accuracy", "Inner accuracy", "Mean inner animal accuracy", "Precision (OIM)", "Recall (OIM)", "F1 (OIM)"}
    for name, frame in tables:
        ws = wb.create_sheet(name); ws.freeze_panes = "A2"
        for i, column in enumerate(frame.columns, 1):
            from openpyxl.utils import get_column_letter
            sample_text = frame[column].dropna().astype(str).head(1000)
            longest = max([len(str(column))] + sample_text.map(len).tolist())
            ws.column_dimensions[get_column_letter(i)].width = min(60, max(18, longest+2))
        ws.auto_filter.ref = f"A1:{get_column_letter(len(frame.columns))}{len(frame)+1}"
        header = []
        for value in frame.columns:
            cell = WriteOnlyCell(ws, value=str(value)); cell.font = Font(bold=True, color="000000")
            cell.fill = PatternFill("solid", fgColor="E8EDF2"); cell.alignment = Alignment(wrap_text=True, vertical="center")
            header.append(cell)
        ws.append(header)
        for row in frame.itertuples(index=False, name=None):
            cells = []
            for column, value in zip(frame.columns, row):
                if isinstance(value, (np.generic,)):
                    value = value.item()
                if value is pd.NA or value is None or (isinstance(value, float) and not np.isfinite(value)):
                    value = None
                cell = WriteOnlyCell(ws, value=value)
                # Animal IDs and notes are data, never executable spreadsheet formulas.
                if isinstance(value, str):
                    cell.data_type = "s"
                if column in percent_names and isinstance(value, (int, float)):
                    cell.number_format = "0.00%"
                cells.append(cell)
            ws.append(cells)
    output = io.BytesIO(); wb.save(output); return output.getvalue()


def result_bundle(result, workbook=None):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("Reviewer1_Comment2_Results.xlsx", workbook if workbook is not None else result_workbook(result))
        for key in ("summary", "folds", "oof", "inner_scores", "component_selection", "input_audit", "selected_rows"):
            archive.writestr(f"{key}.csv", result[key].to_csv(index=False))
        for key in ("manifest", "fitted_configurations", "warnings"):
            archive.writestr(f"{key}.json", json.dumps(_jsonable(result[key]), indent=2, sort_keys=True))
        archive.writestr("Methods.md", methods_text(result))
        archive.writestr("Reviewer_response.md", reviewer_text(result))
        archive.writestr("README.md", "# New Reviewer 1 Comment 2 reanalysis\n\n"
                         "Start with Summary in the Excel workbook. Nested PLS results and fixed-three comparisons are separate. "
                         "OOF CSV contains every held-out prediction. PLS selection and inner-fold records show how components were chosen. "
                         "The manifest and fitted configurations identify this exact run. Upstream historical PMMA/IQR provenance "
                         "was not reconstructed. This is not a certification of other legacy application features.\n")
    return output.getvalue()


def render_panel(st, existing_upload=None, selector=None):
    """Isolated opt-in UI with input/configuration identity checks on every rerun."""
    if not st.sidebar.checkbox("Reviewer 1 Comment 2 — audited validation", value=False, key="r1c2_show"):
        return
    st.header("Reviewer 1 Comment 2: preprocessing and model configuration")
    st.caption("New reanalysis: fixed classifiers, fixed-three PLS comparison and training-only nested PLS component selection.")
    st.info("This run uses the supplied representations without additional PMMA/IQR exclusions or derivative generation. "
            "It cannot recover the workbook's earlier processing history.")
    if existing_upload is not None:
        use_current = st.checkbox("Use the workbook already selected in the application", value=True, key="r1c2_use_current")
    else:
        use_current = False
    uploaded = existing_upload if use_current else (selector("Input workbook", "r1c2_upload") if selector else st.file_uploader("Input workbook", type=["xlsx"], key="r1c2_upload"))
    n = int(st.number_input("Spectra per animal for this reanalysis", 1, 2000, 250, key="r1c2_n"))
    seed = int(st.number_input("Reanalysis seed", 0, 10000, 42, key="r1c2_seed"))
    models = tuple(st.multiselect("Fixed classifiers", MODEL_NAMES, default=list(MODEL_NAMES), key="r1c2_models"))
    notes = st.text_area("Existing preprocessing description, if documented (optional)", key="r1c2_upstream_notes",
                         help="This records an author-supplied description; it does not verify historical exclusions.")
    st.caption("PLS candidates: 1–5. Inner LOGO uses training animals only; the smallest component count wins a tie. "
               "Collagen range remains 1300–1700 cm⁻¹ to preserve the existing implementation; the manuscript's different range needs author resolution.")
    config = AnalysisConfig(spectra_per_animal=n, seed=seed, models=models, upstream_notes=notes)
    if uploaded is None:
        st.info("Select an input workbook to run this analysis.")
        return
    data = uploaded.getvalue()
    fingerprint = run_fingerprint(data, config)
    if st.button("Run audited Comment 2 analysis", type="primary", key="r1c2_run"):
        st.session_state.pop("r1c2_completed", None)
        bar = st.progress(0.0); status = st.empty()
        def update(fraction, message):
            bar.progress(fraction); status.write(message)
        try:
            result = run_analysis(data, config, getattr(uploaded, "name", "uploaded.xlsx"), update)
            workbook = result_workbook(result)
            bundle = result_bundle(result, workbook)
            st.session_state["r1c2_completed"] = {"fingerprint": fingerprint, "result": result,
                                                   "workbook": workbook, "bundle": bundle}
            status.success("Audited reanalysis completed.")
        except Exception as error:
            st.error(f"The analysis was not accepted: {error}")
    completed = st.session_state.get("r1c2_completed")
    if not completed or completed["fingerprint"] != fingerprint:
        st.info("Run this analysis with the current workbook and settings to enable its results and downloads.")
        return
    result = completed["result"]
    st.dataframe(result["summary"], hide_index=True, width="stretch")
    st.subheader("PLS component selection within training animals")
    choices = result["component_selection"]
    st.dataframe(choices[choices["Selected"]], hide_index=True, width="stretch")
    st.caption(f"Completed {result['manifest']['completed_utc']}; input SHA-256 {result['manifest']['input_sha256']}")
    st.download_button("Download Comment 2 results Excel", completed["workbook"], "Reviewer1_Comment2_Results.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="r1c2_excel")
    st.download_button("Download complete reproducibility package", completed["bundle"], "Reviewer1_Comment2_Reanalysis.zip",
                       "application/zip", key="r1c2_bundle")
    st.download_button("Download run-specific Methods", methods_text(result), "Reviewer1_Comment2_Methods.md", "text/markdown", key="r1c2_methods")
    st.download_button("Download reviewer-response text", reviewer_text(result), "Reviewer1_Comment2_Response.md", "text/markdown", key="r1c2_response")
