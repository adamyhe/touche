from __future__ import annotations

import unittest

import touche.api as api


class ApiTests(unittest.TestCase):
    def test_api_exports_notebook_helpers(self) -> None:
        self.assertTrue(callable(api.build_contact_indexes))
        self.assertTrue(callable(api.compute_apa))
        self.assertTrue(callable(api.compute_ep_and_background))
        self.assertTrue(callable(api.compute_local_decay))
        self.assertTrue(callable(api.make_instrumentation))
        self.assertTrue(callable(api.plot_pair_type_distribution))
        self.assertTrue(api.Instrumentation(progress=True).progress)

    def test_api_exports_the_statistics_surface(self) -> None:
        for name in (
            "test_contacts", "assess_calibration", "summarize_apa", "test_background_change",
            "compare_groups", "compare_paired", "correlate", "match_pairs", "balance_table",
            "adjust_pvalues", "bootstrap_ci", "read_loop_calls", "annotate_pairs",
            "build_pair_table", "read_bedpe",
        ):
            self.assertTrue(callable(getattr(api, name)), name)

    def test_every_exported_name_exists_and_is_listed(self) -> None:
        for name in api.__all__:
            self.assertTrue(hasattr(api, name), f"__all__ lists missing name {name}")
        public = {
            name
            for name in vars(api)
            if not name.startswith("_") and getattr(vars(api)[name], "__module__", "").startswith("touche")
        }
        self.assertEqual(public - set(api.__all__), set(), "public names missing from __all__")


if __name__ == "__main__":
    unittest.main()
