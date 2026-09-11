from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import polars as pl

from touche import __version__
from touche.metadata import INFERENCE_CLASSES, METHOD_REGISTRY, StatResult, describe_method, method_info


class MethodRegistryTests(unittest.TestCase):
    def test_every_registered_method_declares_a_known_inference_class(self) -> None:
        for name, entry in METHOD_REGISTRY.items():
            self.assertIn(entry["inference_class"], INFERENCE_CLASSES, name)
            for field in ("hypothesis", "null", "unit_of_analysis", "reference"):
                self.assertTrue(entry[field].strip(), f"{name} is missing {field}")

    def test_legacy_fisher_is_not_advertised_as_calibrated_inference(self) -> None:
        entry = METHOD_REGISTRY["legacy_fisher"]

        self.assertEqual(entry["inference_class"], "descriptive")
        self.assertIn("Not a generative null", entry["null"])

    def test_unregistered_method_gets_an_explicit_undocumented_stub(self) -> None:
        stub = describe_method("something_new")

        self.assertEqual(stub["hypothesis"], "Not documented.")
        self.assertEqual(stub["inference_class"], "descriptive")

    def test_method_info_overrides_registry_fields(self) -> None:
        info = method_info("binomial", alternative="greater", n_input=10, n_tested=8)

        self.assertEqual(info.method, "binomial")
        self.assertEqual(info.inference_class, "technical")
        self.assertEqual(info.alternative, "greater")
        self.assertEqual(info.n_tested, 8)


class StatResultTests(unittest.TestCase):
    def _result(self) -> StatResult:
        return StatResult(
            table=pl.DataFrame({"pair_id": ["chr1:1|chr1:2"], "p_value": [0.01], "q_value": [0.02]}),
            info=method_info("binomial", n_input=1, n_tested=1, fdr_method="bh", fdr_family="all called pairs"),
        )

    def test_write_emits_the_table_and_a_metadata_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._result().write(Path(tmp) / "nested" / "result.tsv")
            table = pl.read_csv(paths["table"], separator="\t")
            metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))

        self.assertEqual(paths["metadata"].name, "result.tsv.meta.json")
        self.assertEqual(table["pair_id"].to_list(), ["chr1:1|chr1:2"])
        self.assertEqual(metadata["method"], "binomial")
        self.assertEqual(metadata["fdr_family"], "all called pairs")
        self.assertEqual(metadata["touche_version"], __version__)
        self.assertEqual(metadata["rows"], 1)

    def test_metadata_dict_is_json_serializable_and_complete(self) -> None:
        payload = self._result().to_dict()

        json.dumps(payload)
        for key in (
            "method", "inference_class", "hypothesis", "null", "alternative",
            "unit_of_analysis", "cluster_unit", "n_input", "n_tested",
            "fdr_method", "fdr_family", "seed", "parameters", "filtered",
            "warnings", "reference", "touche_version", "created_at",
        ):
            self.assertIn(key, payload)


if __name__ == "__main__":
    unittest.main()
