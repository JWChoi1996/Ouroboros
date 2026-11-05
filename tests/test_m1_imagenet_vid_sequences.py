import tempfile
import unittest
from pathlib import Path


class M1ImageNetVidSequenceTests(unittest.TestCase):
    def test_resolves_repeatable_sequences_in_requested_order_and_exact_count(self):
        from on_device.motivation_m1.imagenet_vid_sequences import resolve_sequences

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("seq-b", "seq-a"):
                sequence = root / "data/vid/vid_cocoval/frames" / name
                sequence.mkdir(parents=True)
                for index in range(4):
                    (sequence / f"{index:06d}.jpg").touch()

            resolved = resolve_sequences(root, ["seq-b", "seq-a"], frame_count=3)

        self.assertEqual(list(resolved), ["seq-b", "seq-a"])
        self.assertEqual(
            [[path.name for path in paths] for paths in resolved.values()],
            [
                ["000000.jpg", "000001.jpg", "000002.jpg"],
                ["000000.jpg", "000001.jpg", "000002.jpg"],
            ],
        )

    def test_rejects_duplicate_or_more_than_three_sequences(self):
        from on_device.motivation_m1.imagenet_vid_sequences import resolve_sequences

        with self.assertRaisesRegex(ValueError, "2 or 3 unique"):
            resolve_sequences(Path("."), ["same", "same"], 30)
        with self.assertRaisesRegex(ValueError, "2 or 3 unique"):
            resolve_sequences(Path("."), ["a", "b", "c", "d"], 30)


if __name__ == "__main__":
    unittest.main()
