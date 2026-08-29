from __future__ import annotations

from torch.utils.data import DataLoader

from lcrseg.data import H5LabeledDataset, H5UnlabeledDataset, WeakStrongTransform, collate_labeled, collate_unlabeled

from .conftest import make_synthetic_root


def test_h5_worker_reading_zero_and_two_workers(tmp_path) -> None:
    root = make_synthetic_root(tmp_path, records=6)
    labeled = H5LabeledDataset(root, seed=0, dataset="fundus")
    unlabeled = H5UnlabeledDataset(root, seed=0, dataset="fundus", transform=WeakStrongTransform(flip_probability=0.0, cutout_probability=0.0))
    for workers in (0, 2):
        labeled_batches = list(DataLoader(labeled, batch_size=1, num_workers=workers, collate_fn=collate_labeled))
        unlabeled_batches = list(DataLoader(unlabeled, batch_size=1, num_workers=workers, collate_fn=collate_unlabeled))
        assert labeled_batches and unlabeled_batches
        assert all(not hasattr(batch, "label") for batch in unlabeled_batches)
