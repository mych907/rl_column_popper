import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Any


def read_metrics(path: Path) -> Tuple[List[int], List[float], List[float]]:
    steps: List[int] = []
    ent: List[float] = []
    rew: List[float] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                steps.append(int(row["total_timesteps"]))
            except Exception:
                continue
            # Empty string means missing value at that snapshot
            e = row.get("entropy_loss", "")
            r = row.get("eval_mean_reward", "")
            ent.append(float(e) if e not in (None, "") else float("nan"))
            rew.append(float(r) if r not in (None, "") else float("nan"))
    return steps, ent, rew


def read_progress(progress_path: Path) -> Tuple[List[int], Dict[str, List[float]]]:
    steps: List[int] = []
    series: Dict[str, List[float]] = {}
    if not progress_path.exists():
        return steps, series
    with open(progress_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                t = int(float(row.get("time/total_timesteps", "")))
            except Exception:
                continue
            steps.append(t)
            # collect a subset of useful keys if present
            keys = [
                "train/entropy_loss",
                "train/value_loss",
                "train/policy_gradient_loss",
                "train/approx_kl",
                "train/clip_fraction",
                "rollout/ep_rew_mean",
                "rollout/ep_len_mean",
            ]
            for k in keys:
                v = row.get(k)
                try:
                    fv = float(v) if v not in (None, "") else float("nan")
                except Exception:
                    fv = float("nan")
                series.setdefault(k, []).append(fv)
    return steps, series


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training metrics (entropy loss and eval reward)")
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("models/ppo_column_popper_metrics.csv"),
        help="Path to metrics CSV produced during training",
    )
    parser.add_argument(
        "--progress",
        type=Path,
        default=Path("models/logs/progress.csv"),
        help="Path to SB3 progress.csv (optional)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("images/metrics.png"),
        help="Output image path",
    )
    args = parser.parse_args()

    steps, ent, rew = read_metrics(args.metrics)
    if not steps:
        raise SystemExit(f"No data found in {args.metrics}")

    prog_steps, prog_series = read_progress(args.progress)

    # Plot using matplotlib
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import numpy as np  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit("matplotlib and numpy are required. Install with: pip install matplotlib numpy")

    steps_arr = np.array(steps)
    ent_arr = np.array(ent, dtype=float)
    rew_arr = np.array(rew, dtype=float)

    # Create a 2x3 grid of plots
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.ravel()

    # Panel 1: Rewards (eval vs. rollout)
    ax = axes[0]
    ax.set_title("Reward vs Timesteps")
    ax.plot(steps_arr, rew_arr, label="eval mean reward", color="tab:blue")
    if prog_steps and "rollout/ep_rew_mean" in prog_series:
        ax.plot(np.array(prog_steps), np.array(prog_series["rollout/ep_rew_mean"]),
                label="rollout ep_rew_mean", color="tab:green", alpha=0.8)
    ax.set_xlabel("timesteps")
    ax.set_ylabel("reward")
    ax.legend(loc="best")

    # Panel 2: Entropy loss
    ax = axes[1]
    ax.set_title("Entropy Loss")
    ax.plot(steps_arr, ent_arr, label="entropy loss (train)", color="tab:orange")
    if prog_steps and "train/entropy_loss" in prog_series:
        ax.plot(np.array(prog_steps), np.array(prog_series["train/entropy_loss"]),
                label="SB3 entropy loss", color="tab:red", alpha=0.6)
    ax.set_xlabel("timesteps")
    ax.legend(loc="best")

    # Panel 3: Value loss
    ax = axes[2]
    ax.set_title("Value Loss")
    if prog_steps and "train/value_loss" in prog_series:
        ax.plot(np.array(prog_steps), np.array(prog_series["train/value_loss"]), color="tab:purple")
    ax.set_xlabel("timesteps")

    # Panel 4: Policy gradient loss
    ax = axes[3]
    ax.set_title("Policy Gradient Loss")
    if prog_steps and "train/policy_gradient_loss" in prog_series:
        ax.plot(np.array(prog_steps), np.array(prog_series["train/policy_gradient_loss"]), color="tab:brown")
    ax.set_xlabel("timesteps")

    # Panel 5: KL and Clip Fraction (twin axes)
    ax = axes[4]
    ax.set_title("Approx KL & Clip Fraction")
    if prog_steps:
        if "train/approx_kl" in prog_series:
            ax.plot(np.array(prog_steps), np.array(prog_series["train/approx_kl"]), color="tab:olive", label="approx_kl")
        ax2 = ax.twinx()
        if "train/clip_fraction" in prog_series:
            ax2.plot(np.array(prog_steps), np.array(prog_series["train/clip_fraction"]), color="tab:pink", label="clip_fraction")
        ax.set_xlabel("timesteps")
        # combine legends
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="best")

    # Panel 6: Episode length
    ax = axes[5]
    ax.set_title("Episode Length (rollout)")
    if prog_steps and "rollout/ep_len_mean" in prog_series:
        ax.plot(np.array(prog_steps), np.array(prog_series["rollout/ep_len_mean"]), color="tab:cyan")
    ax.set_xlabel("timesteps")

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()
