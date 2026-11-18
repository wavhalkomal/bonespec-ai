import hashlib
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import warnings
import zipfile

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler as RealScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reviewer1_comment2 as engine


def fixture(rows=10):
    rng = np.random.RandomState(17)
    columns = [1800, 1760, 1700, 1650, 1600, 1550, 1500, 1400, 1300, 1200, 1100, 1050, 1030, 1000, 900, 800]
    records = []
    for i in range(6):
        label = "WT" if i < 3 else "OIM"
        for j in range(rows):
            values = rng.normal(size=len(columns)) + (i >= 3)*.35
            records.append({"Sample ID": f"New_{i}", "Type": label, **dict(zip(columns, values))})
    return pd.DataFrame(records)


def small_result(models=()):
    config = engine.AnalysisConfig(spectra_per_animal=6, candidates=(1, 2), models=models)
    views, audit, selected = engine.prepare_views({"Raw": fixture()}, config)
    # One real feature set is sufficient for the unit-level isolation checks.
    with engine.threadpool_limits(limits=2):
        return engine.analyze_views(views[:1], audit, selected, config, "fixture-hash"), views[:1], audit, selected, config


class InputChecks(unittest.TestCase):
    def test_dynamic_ids_sampling_and_source_rows(self):
        frame = fixture()
        config = engine.AnalysisConfig(spectra_per_animal=6, candidates=(1, 2), models=())
        views, audit, selected = engine.prepare_views({"Raw": frame}, config)
        self.assertEqual(set(selected.Animal), set(frame["Sample ID"]))
        self.assertTrue((selected.groupby("Animal").size() == 6).all())
        self.assertTrue((audit["Additional QC exclusions"] == 0).all())
        source = frame.iloc[selected["Source Excel row"].to_numpy()-2]
        np.testing.assert_array_equal(source["Sample ID"], selected.Animal)
        expected = pd.concat([g.sample(n=6, random_state=42) for _, g in frame.groupby("Sample ID", sort=True)])
        np.testing.assert_array_equal(expected.index+2, selected["Source Excel row"])
        np.testing.assert_array_equal(views[0]["y"], (source.Type == "WT").astype(int))

    def test_invalid_values_rejected_with_source_location(self):
        for bad in (np.nan, np.inf, "invalid"):
            with self.subTest(value=bad):
                frame = fixture(); frame[1800] = frame[1800].astype(object); frame.loc[0, 1800] = bad
                with self.assertRaisesRegex(ValueError, "Excel row 2"):
                    engine.prepare_views({"Raw": frame}, engine.AnalysisConfig(spectra_per_animal=6))

    def test_conflicting_phenotype_rejected_before_sampling(self):
        frame = fixture(); frame.loc[0, "Type"] = "OIM"
        with self.assertRaisesRegex(ValueError, "conflicting phenotypes"):
            engine.prepare_views({"Raw": frame}, engine.AnalysisConfig(spectra_per_animal=6))

    def test_shortfalls_do_not_fall_back_to_unequal_counts(self):
        with self.assertRaisesRegex(ValueError, "no unequal fallback"):
            engine.prepare_views({"Raw": fixture(3)}, engine.AnalysisConfig(spectra_per_animal=4))

    def test_cross_sheet_cohorts_must_match(self):
        raw = fixture(); derivative = raw.copy(); derivative["Sample ID"] = derivative["Sample ID"].replace({"New_0": "Different"})
        with self.assertRaisesRegex(ValueError, "differ between"):
            engine.prepare_views({"Raw": raw, "2nd derivative": derivative}, engine.AnalysisConfig(spectra_per_animal=6))

    def test_nested_selection_requires_enough_animals(self):
        frame = fixture(); frame = frame[~frame["Sample ID"].isin(["New_0", "New_3"])]
        with self.assertRaisesRegex(ValueError, "at least three"):
            engine.prepare_views({"Raw": frame}, engine.AnalysisConfig(spectra_per_animal=6))


class NestedValidationChecks(unittest.TestCase):
    def test_inner_scalers_never_fit_validation_or_outer_test_rows(self):
        groups = np.repeat(np.arange(6), 6)
        y = (groups >= 3).astype(int)
        rng = np.random.RandomState(3)
        X = rng.normal(size=(36, 5)); X[:, 0] = np.arange(36)
        outer = groups != 0
        fits = []
        class RecordingScaler:
            def fit(self, values):
                fits.append(set(values[:, 0].astype(int)))
                self.real = RealScaler().fit(values)
                return self
            def transform(self, values):
                return self.real.transform(values)
        with patch.object(engine, "StandardScaler", RecordingScaler):
            _, inner, _ = engine.select_pls_components(X[outer], y[outer], groups[outer], (1, 2), "0")
        expected = [set(np.flatnonzero(outer & (groups != animal))) for animal in sorted(set(groups[outer]))]
        self.assertEqual(fits, expected)
        self.assertTrue(all(not s.intersection(set(np.flatnonzero(~outer))) for s in fits))
        self.assertTrue(all("0" not in json.loads(s) for s in inner["Inner training animals"]))

    def test_outer_test_feature_changes_do_not_change_its_component_selection(self):
        original, views, audit, selected, config = small_result()
        altered = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in views[0].items()}
        animal = sorted(set(altered["groups"]))[0]
        altered["X"][altered["groups"] == animal] += 1e7
        with engine.threadpool_limits(limits=2):
            changed = engine.analyze_views([altered], audit, selected, config, "changed-test-data")
        old_scores = original["inner_scores"].query('`Outer held-out animal` == @animal').reset_index(drop=True)
        new_scores = changed["inner_scores"].query('`Outer held-out animal` == @animal').reset_index(drop=True)
        pd.testing.assert_frame_equal(old_scores, new_scores)
        old_hashes = {x["Scaler mean sha256"] for x in original["fitted_configurations"] if x["Animal"] == animal}
        new_hashes = {x["Scaler mean sha256"] for x in changed["fitted_configurations"] if x["Animal"] == animal}
        self.assertEqual(old_hashes, new_hashes)

    def test_component_tie_chooses_smallest_and_grid_is_dimension_bounded(self):
        groups = np.repeat(np.arange(6), 5); y = (groups >= 3).astype(int)
        signal = y.astype(float)
        X = np.column_stack([signal, signal*2, signal*3])
        chosen, _, choices = engine.select_pls_components(X, y, groups, (1, 2, 3, 4), "outer")
        self.assertEqual(chosen, 1)
        self.assertEqual(set(choices.Components), {1, 2, 3})
        self.assertTrue((choices["Mean inner animal accuracy"] == 1).all())

    def test_convergence_warning_prevents_accepted_run(self):
        class NotConverged:
            def fit(self, X, y):
                warnings.warn("did not converge", ConvergenceWarning)
        records = []
        with self.assertRaisesRegex(ValueError, "did not converge"):
            engine._fit(NotConverged(), np.zeros((5, 3)), np.arange(5), {"Stage": "test"}, records)
        self.assertEqual(records[0]["Category"], "ConvergenceWarning")

    def test_oof_coverage_metrics_and_positive_phenotype(self):
        with patch.object(engine, "build_models", return_value={"SVM": DummyClassifier(strategy="constant", constant=0)}):
            result, _, _, _, _ = small_result(("SVM",))
        p = result["oof"].query('Model == "SVM"')
        self.assertEqual(len(p), 36)
        self.assertFalse(p.duplicated(["Animal", "Source Excel row"]).any())
        self.assertEqual(set(p["Predicted phenotype"]), {"OIM"})
        row = result["summary"].query('Model == "SVM"').iloc[0]
        self.assertEqual(row["OOF accuracy"], .5)
        self.assertEqual(row["Precision (OIM)"], .5)
        self.assertEqual(row["Recall (OIM)"], 1.)
        self.assertTrue((result["folds"]["Test rows"] == 6).all())


class ExportChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result, _, _, _, _ = small_result()

    def test_excel_values_match_computed_results_and_contain_provenance(self):
        blob = engine.result_workbook(self.result)
        wb = load_workbook(io.BytesIO(blob), read_only=True, data_only=True)
        self.assertEqual(len(wb.sheetnames), 7)
        rows = list(wb["Summary"].values)
        headers = rows[0]
        for saved, computed in zip(rows[1:], self.result["summary"].itertuples(index=False, name=None)):
            for a, b in zip(saved, computed):
                if isinstance(b, (float, np.floating)):
                    self.assertAlmostEqual(a, b)
                else:
                    self.assertEqual(a, b)
        self.assertIn("OOF accuracy", headers)
        self.assertTrue(any(r[1] == "historical_PMMA_IQR_reconstructed" and r[2] is False for r in wb["Run configuration"].values))
        wb.close()

    def test_untrusted_ids_are_text_in_excel(self):
        result = dict(self.result); result["selected_rows"] = self.result["selected_rows"].copy()
        result["selected_rows"].loc[0, "Animal"] = "=SUM(1,2)"
        wb = load_workbook(io.BytesIO(engine.result_workbook(result)), read_only=True, data_only=False)
        cells = next(wb["Selected spectra"].iter_rows(min_row=2, max_row=2))
        self.assertEqual(cells[2].value, "=SUM(1,2)")
        self.assertEqual(cells[2].data_type, "s")
        wb.close()

    def test_package_contains_complete_oof_and_runtime_records(self):
        with zipfile.ZipFile(io.BytesIO(engine.result_bundle(self.result))) as z:
            self.assertIn("oof.csv", z.namelist())
            self.assertIn("fitted_configurations.json", z.namelist())
            self.assertIn("manifest.json", z.namelist())
            p = pd.read_csv(z.open("oof.csv"))
            self.assertEqual(len(p), len(self.result["oof"]))
            self.assertIn("conditional on the supplied dataset", z.read("Methods.md").decode())

    def test_run_identity_changes_with_workbook_or_configuration(self):
        config = engine.AnalysisConfig()
        a = engine.run_fingerprint(b"workbook A", config)
        self.assertNotEqual(a, engine.run_fingerprint(b"workbook B", config))
        self.assertNotEqual(a, engine.run_fingerprint(b"workbook A", engine.AnalysisConfig(seed=43)))
        self.assertNotEqual(a, engine.run_fingerprint(b"workbook A", engine.AnalysisConfig(candidates=(1, 2))))

    def test_working_app_is_byte_identical(self):
        updated = Path(__file__).resolve().parents[1].joinpath("app.py").read_bytes()
        git_blob = hashlib.sha1(b"blob "+str(len(updated)).encode()+b"\0"+updated).hexdigest()
        self.assertEqual(git_blob, "0e67a5bf9ab52c98c0f2296544da30a8595827bd")
        self.assertTrue(Path(__file__).resolve().parents[1].joinpath("pages/2_Reviewer1_Comment2.py").is_file())


if __name__ == "__main__":
    unittest.main()
