import numpy as np
import pandas as pd
import pytest

from blam_analysis.data import build_stats, compute_sampling_stats


def test_compute_sampling_stats_matches_hand_computation():
    A = pd.DataFrame({
        "m1": [1.0, 2.0, 3.0, 4.0],
        "m2": [2.0, 2.0, 2.0, 2.0],
        "m3": [0.0, 1.0, 4.0, 5.0],
    })
    y_hat, V, n_pairs = compute_sampling_stats(A)

    expected_y_hat = A.mean(axis=0).to_numpy()
    assert np.allclose(y_hat, expected_y_hat)

    X = A.to_numpy()
    n = X.shape[0]
    expected_cov = np.cov(X, rowvar=False, ddof=1) / n
    assert np.allclose(V, expected_cov)
    assert (n_pairs == n).all()


def test_build_stats_from_long_dataframe():
    df = pd.DataFrame([
        {"model": "m1", "task": "t1", "sample_id": "s1", "y_true": 1.0, "y_pred": 1.5},
        {"model": "m1", "task": "t1", "sample_id": "s2", "y_true": 2.0, "y_pred": 1.0},
        {"model": "m1", "task": "t1", "sample_id": "s3", "y_true": 3.0, "y_pred": 3.5},
        {"model": "m2", "task": "t1", "sample_id": "s1", "y_true": 1.0, "y_pred": 2.0},
        {"model": "m2", "task": "t1", "sample_id": "s2", "y_true": 2.0, "y_pred": 2.5},
        {"model": "m2", "task": "t1", "sample_id": "s3", "y_true": 3.0, "y_pred": 2.0},
        {"model": "m2", "task": "t1", "sample_id": "s4", "y_true": 4.0, "y_pred": 4.5},
    ])
    stats = build_stats(df)
    assert len(stats) == 1
    stat = stats[0]
    assert stat.name == "t1"
    assert stat.models == ["m1", "m2"]
    assert stat.n_samples == 4

    m1_errors = np.array([0.5, 1.0, 0.5])
    m2_errors = np.array([1.0, 0.5, 1.0, 0.5])
    assert np.isclose(stat.y_hat[1], m2_errors.mean())
    assert np.isclose(stat.y_hat[0], m1_errors.mean())


def test_build_stats_skips_task_with_too_few_models():
    df = pd.DataFrame([
        {"model": "m1", "task": "t1", "sample_id": "s1", "y_true": 1.0, "y_pred": 1.5},
        {"model": "m1", "task": "t1", "sample_id": "s2", "y_true": 2.0, "y_pred": 1.0},
        {"model": "m1", "task": "t2", "sample_id": "s1", "y_true": 1.0, "y_pred": 1.1},
        {"model": "m1", "task": "t2", "sample_id": "s2", "y_true": 2.0, "y_pred": 2.2},
        {"model": "m2", "task": "t2", "sample_id": "s1", "y_true": 1.0, "y_pred": 1.2},
        {"model": "m2", "task": "t2", "sample_id": "s2", "y_true": 2.0, "y_pred": 1.8},
    ])
    stats = build_stats(df)
    assert [s.name for s in stats] == ["t2"]


def test_build_stats_raises_on_duplicate_rows():
    df = pd.DataFrame([
        {"model": "m1", "task": "t1", "sample_id": "s1", "y_true": 1.0, "y_pred": 1.5},
        {"model": "m1", "task": "t1", "sample_id": "s1", "y_true": 1.0, "y_pred": 1.6},
        {"model": "m2", "task": "t1", "sample_id": "s1", "y_true": 1.0, "y_pred": 1.2},
    ])
    with pytest.raises(ValueError, match="duplicate"):
        build_stats(df)


def test_compute_sampling_stats_raises_on_too_few_shared_samples():
    A = pd.DataFrame({
        "m1": [1.0, np.nan, np.nan],
        "m2": [np.nan, 1.0, 2.0],
    })
    with pytest.raises(ValueError, match="share fewer than 2"):
        compute_sampling_stats(A)
