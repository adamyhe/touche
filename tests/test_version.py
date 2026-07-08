from __future__ import annotations

import unittest
from importlib.metadata import version

import touche


class VersionTests(unittest.TestCase):
    def test_package_version_matches_distribution_metadata(self) -> None:
        self.assertEqual(touche.__version__, version("ep-touche"))


if __name__ == "__main__":
    unittest.main()
