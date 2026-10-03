from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.splits import create_split_manifest, sha256_file


class SplitManifestTests(unittest.TestCase):
    def test_split_is_stratified_and_duplicate_groups_stay_together(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            train_root = root / "data" / "train"
            for class_name in ("A", "B"):
                class_dir = train_root / class_name
                class_dir.mkdir(parents=True)
                for index in range(10):
                    payload = f"{class_name}-{index}".encode()
                    if class_name == "A" and index == 9:
                        payload = b"A-8"
                    (class_dir / f"{index}.jpg").write_bytes(payload)

            manifest = create_split_manifest(
                train_root=train_root,
                output_path=root / "splits" / "test.json",
                project_root=root,
                val_fraction=0.30,
                seed=7,
            )

            self.assertEqual(manifest["train_count"], 14)
            self.assertEqual(manifest["val_count"], 6)
            for counts in manifest["class_counts"].values():
                self.assertEqual(counts, {"total": 10, "train": 7, "val": 3})

            assignments = {path: "train" for path in manifest["train_paths"]}
            assignments.update({path: "val" for path in manifest["val_paths"]})
            duplicate_group = manifest["duplicate_groups"][0]
            self.assertEqual({assignments[path] for path in duplicate_group}, {assignments[duplicate_group[0]]})

    def test_conflicting_cross_class_duplicate_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            train_root = root / "data" / "train"
            for class_name in ("A", "B"):
                class_dir = train_root / class_name
                class_dir.mkdir(parents=True)
                (class_dir / "same.jpg").write_bytes(b"identical")
                (class_dir / "unique.jpg").write_bytes(class_name.encode())

            with self.assertRaisesRegex(ValueError, "conflicting labels"):
                create_split_manifest(
                    train_root=train_root,
                    output_path=root / "split.json",
                    project_root=root,
                    val_fraction=0.50,
                    seed=0,
                )

    def test_sha256_is_content_based(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = root / "first.jpg"
            second = root / "second.jpg"
            first.write_bytes(b"same bytes")
            second.write_bytes(b"same bytes")
            self.assertEqual(sha256_file(first), sha256_file(second))


if __name__ == "__main__":
    unittest.main()

