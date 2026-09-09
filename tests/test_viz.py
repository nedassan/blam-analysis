import os

import pandas as pd
import pytest

pytest.importorskip("matplotlib")

from blam_analysis import viz  # noqa: E402


@pytest.fixture
def theta_df():
    return pd.DataFrame({
        "model": ["m1", "m2", "m3"],
        "theta_mean": [0.5, -0.2, -0.3],
        "theta_std": [0.1, 0.1, 0.1],
    })


@pytest.fixture
def spec_df():
    return pd.DataFrame({
        "task": ["t1", "t1", "t2", "t2"],
        "model": ["m1", "m2", "m1", "m2"],
        "u_mean": [0.1, -0.1, -0.2, 0.2],
        "u_std": [0.05, 0.05, 0.05, 0.05],
    })


@pytest.fixture
def task_df():
    return pd.DataFrame({
        "task": ["t1", "t2"],
        "mu_mean": [1.0, 1.5],
        "mu_std": [0.1, 0.1],
        "beta_mean": [1.0, 1.0],
        "beta_std": [0.1, 0.1],
        "omega_mean": [0.3, 0.3],
        "omega_std": [0.05, 0.05],
        "n_samples": [10, 10],
        "n_models": [2, 2],
    })


@pytest.fixture
def comparison_df():
    return pd.DataFrame({
        "model": ["m2", "m3"],
        "prob_beat_baseline": [0.4, 0.6],
        "p_value": [0.2, 0.03],
        "delta_mean": [-0.1, 0.1],
    })


def _assert_nonempty(path):
    assert os.path.exists(path)
    assert os.path.getsize(path) > 0


def test_plot_theta(tmp_path, theta_df):
    output = str(tmp_path / "theta.png")
    viz.plot_theta(theta_df, output)
    _assert_nonempty(output)


def test_plot_comparison(tmp_path, comparison_df):
    output = str(tmp_path / "comparison.png")
    viz.plot_comparison(comparison_df, output, baseline="m1", alpha=0.05)
    _assert_nonempty(output)


def test_write_dashboard_html(tmp_path, theta_df, spec_df, task_df):
    output = str(tmp_path / "dashboard.html")
    viz.write_dashboard_html(theta_df, spec_df, task_df, output)
    _assert_nonempty(output)


def test_write_task_comparison_html(tmp_path, comparison_df, task_df):
    per_task_df = pd.DataFrame({
        "task": ["t1", "t1", "t2", "t2"],
        "model": ["m2", "m3", "m2", "m3"],
        "prob_beat_baseline": [0.3, 0.7, 0.4, 0.6],
        "p_value": [0.1, 0.04, 0.2, 0.06],
    })
    output = str(tmp_path / "task_comparison.html")
    viz.write_task_comparison_html(per_task_df, task_df, baseline="m1", alpha=0.05, output=output, new_task_df=comparison_df)
    _assert_nonempty(output)
