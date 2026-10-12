import numpy as np
import pytest

from vae_wwx.data.read_real_emg import RealEMGDataset


def test_shared_scaler_keeps_force_difference_without_validation_leakage(tmp_path):
    base = 2 + np.sin(np.arange(16, dtype=np.float32))
    for name, force in (("weak", 1.0), ("strong", 3.0), ("validation", 20.0)):
        signal = np.tile((base * force)[:, None], (1, 2))
        labels = np.ones((len(base), 1), dtype=np.float32)
        np.savetxt(tmp_path / f"{name}_labeled.csv", np.column_stack((signal, labels)), delimiter=",")

    dataset = RealEMGDataset(tmp_path, window_size=8, stride=8, n_channels=2, n_classes=2)
    ids = {name: i for i, name in enumerate(dataset.group_names)}
    train_groups = {ids["weak_labeled.csv"], ids["strong_labeled.csv"]}
    train_indices = [i for i, group in enumerate(dataset.group_ids.tolist()) if group in train_groups]
    expected_mean = dataset.x[train_indices].mean(dim=(0, 2)).numpy()
    raw_rms = dataset.emg_rms.clone()

    stats = dataset.standardize_from_train(train_indices)

    np.testing.assert_allclose(stats["mean"], expected_mean, rtol=1e-6)
    np.testing.assert_allclose(dataset.emg_rms.numpy(), raw_rms.numpy())
    mean_by_file = {
        name: dataset.x[dataset.group_ids == group].mean().item()
        for name, group in ids.items()
    }
    assert mean_by_file["weak_labeled.csv"] < mean_by_file["strong_labeled.csv"]
    assert mean_by_file["strong_labeled.csv"] < mean_by_file["validation_labeled.csv"]
    with pytest.raises(RuntimeError):
        dataset.standardize_from_train(train_indices)

