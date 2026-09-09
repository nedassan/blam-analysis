import os

import joblib
import pandas as pd


class BLAMResult:
    def __init__(
        self, samples, grouped_samples, stats, u_by_task, global_models,
        theta_summary, specialization_summary, task_summary,
    ):
        self.samples = samples
        self.grouped_samples = grouped_samples
        self.stats = stats
        self.u_by_task = u_by_task
        self.global_models = global_models
        self.theta_summary = theta_summary
        self.specialization_summary = specialization_summary
        self.task_summary = task_summary

    def compare(self, baseline: str, seed: int = 0) -> pd.DataFrame:
        from blam_analysis.comparison import compare_to_baseline

        return compare_to_baseline(self, baseline, seed)

    def compare_per_task(self, baseline: str) -> pd.DataFrame:
        from blam_analysis.comparison import compare_to_baseline_per_task

        return compare_to_baseline_per_task(self, baseline)

    def mae_summary(self) -> pd.DataFrame:
        from blam_analysis.comparison import compute_mae_summary

        return compute_mae_summary(self)

    def save(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(
            {
                "samples": self.samples,
                "grouped_samples": self.grouped_samples,
                "stats": self.stats,
                "u_by_task": self.u_by_task,
                "global_models": self.global_models,
            },
            os.path.join(output_dir, "posterior.joblib"),
        )
        self.theta_summary.to_csv(os.path.join(output_dir, "theta_summary.csv"), index=False)
        self.specialization_summary.to_csv(os.path.join(output_dir, "specialization_summary.csv"), index=False)
        self.task_summary.to_csv(os.path.join(output_dir, "task_summary.csv"), index=False)

    @classmethod
    def load(cls, output_dir: str) -> "BLAMResult":
        posterior = joblib.load(os.path.join(output_dir, "posterior.joblib"))
        theta_summary = pd.read_csv(os.path.join(output_dir, "theta_summary.csv"))
        specialization_summary = pd.read_csv(os.path.join(output_dir, "specialization_summary.csv"))
        task_summary = pd.read_csv(os.path.join(output_dir, "task_summary.csv"))
        return cls(
            samples=posterior["samples"],
            grouped_samples=posterior.get("grouped_samples"),
            stats=posterior["stats"],
            u_by_task=posterior["u_by_task"],
            global_models=posterior["global_models"],
            theta_summary=theta_summary,
            specialization_summary=specialization_summary,
            task_summary=task_summary,
        )
