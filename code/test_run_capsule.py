import unittest
from unittest.mock import patch

from run_capsule import ensure_was_generated_by


class FakeNWBFile:
    def __init__(self, was_generated_by):
        self.fields = {"was_generated_by": was_generated_by}

    @property
    def was_generated_by(self):
        return self.fields.get("was_generated_by")

    @was_generated_by.setter
    def was_generated_by(self, value):
        if "was_generated_by" in self.fields:
            raise AttributeError("was_generated_by is already set")
        self.fields["was_generated_by"] = value


class EnsureWasGeneratedByTests(unittest.TestCase):
    @patch("run_capsule.package_version", return_value="0.2.5")
    def test_replaces_empty_provenance_with_installed_package_version(
        self, package_version
    ):
        nwbfile = FakeNWBFile([])

        ensure_was_generated_by(nwbfile)

        self.assertEqual(
            nwbfile.was_generated_by,
            [["aind-nwb-utils", "0.2.5"]],
        )
        package_version.assert_called_once_with("aind-nwb-utils")

    @patch("run_capsule.package_version")
    def test_preserves_existing_provenance(self, package_version):
        provenance = [["existing-package", "1.0.0"]]
        nwbfile = FakeNWBFile(provenance)

        ensure_was_generated_by(nwbfile)

        self.assertIs(nwbfile.was_generated_by, provenance)
        package_version.assert_not_called()


if __name__ == "__main__":
    unittest.main()