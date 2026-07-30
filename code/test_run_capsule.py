import unittest
from unittest.mock import patch

from run_capsule import ensure_was_generated_by, find_eye_tracking_paths


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


class FindEyeTrackingPathsTests(unittest.TestCase):
    def test_prefers_dedicated_eye_tracking_asset(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            root = Path(directory)
            eye_path = root / "eye_tracking"
            processed_path = root / "processed"
            eye_path.mkdir()
            processed_path.mkdir()
            dedicated = eye_path / "ellipses_processed.h5"
            fallback = processed_path / "ellipses_processed_old.h5"
            dedicated.touch()
            fallback.touch()

            self.assertEqual(
                find_eye_tracking_paths(eye_path, processed_path), [dedicated]
            )

    def test_falls_back_to_processed_asset(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            root = Path(directory)
            fallback = root / "processed" / "ellipses_processed.h5"
            fallback.parent.mkdir()
            fallback.touch()

            self.assertEqual(
                find_eye_tracking_paths(root / "missing", fallback.parent),
                [fallback],
            )


if __name__ == "__main__":
    unittest.main()