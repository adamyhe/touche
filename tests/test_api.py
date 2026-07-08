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


if __name__ == "__main__":
    unittest.main()
