"""
Plot training curves from epoch_metrics.json produced by train_qwen3.py.

Outputs (saved to --output_dir):
  loss_curve.png       — train loss vs eval loss
  accuracy_curve.png   — eval accuracy (boxed-answer exact match)
  combined_curves.png  — both panels in one figure
"""

import json
import argparse
import os
import sys

try:
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend, safe on headless servers
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
except ImportError:
    sys.exit("matplotlib is required:  pip install matplotlib")


def load_metrics(path: str):
    with open(path) as f:
        data = json.load(f)
    epochs      = [d["epoch"]          for d in data]
    train_loss  = [d["train_loss"]      for d in data]
    eval_loss   = [d["eval_loss"]       for d in data]
    accuracy    = [d["eval_accuracy"] * 100 for d in data]   # percent
    return epochs, train_loss, eval_loss, accuracy


def style_ax(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_loss(epochs, train_loss, eval_loss, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, "b-o", linewidth=2, markersize=6, label="Train loss")
    ax.plot(epochs, eval_loss,  "r-o", linewidth=2, markersize=6, label="Eval loss")
    style_ax(ax, "Training & Evaluation Loss", "Epoch", "Loss")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_accuracy(epochs, accuracy, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, accuracy, "g-o", linewidth=2, markersize=6, label="Eval accuracy")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    style_ax(ax, r"Eval Accuracy ($\boxed{}$ exact match)", "Epoch", "Accuracy (%)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_combined(epochs, train_loss, eval_loss, accuracy, save_path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 10), sharex=True)

    ax1.plot(epochs, train_loss, "b-o", linewidth=2, markersize=6, label="Train loss")
    ax1.plot(epochs, eval_loss,  "r-o", linewidth=2, markersize=6, label="Eval loss")
    style_ax(ax1, "Loss Curves", "Epoch", "Loss")

    ax2.plot(epochs, accuracy, "g-o", linewidth=2, markersize=6, label="Eval accuracy")
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    style_ax(ax2, r"Eval Accuracy ($\boxed{}$ exact match)", "Epoch", "Accuracy (%)")

    fig.suptitle("Qwen3-3B LoRA — NuminaMath-CoT Finetuning", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics_file", type=str,
        default="./output/qwen3-3b-numinamath/epoch_metrics.json",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="./output/qwen3-3b-numinamath/plots",
    )
    args = parser.parse_args()

    if not os.path.exists(args.metrics_file):
        sys.exit(f"Metrics file not found: {args.metrics_file}")

    epochs, train_loss, eval_loss, accuracy = load_metrics(args.metrics_file)

    if len(epochs) == 0:
        sys.exit("No epochs found in metrics file — training may not have completed an epoch yet.")

    # Filter out None entries (e.g., if eval_loss wasn't available for an epoch)
    valid = [(e, tl, el, a) for e, tl, el, a in zip(epochs, train_loss, eval_loss, accuracy)
             if tl is not None and el is not None]
    if not valid:
        sys.exit("All metric entries have None values — nothing to plot.")
    epochs, train_loss, eval_loss, accuracy = map(list, zip(*valid))

    os.makedirs(args.output_dir, exist_ok=True)

    plot_loss(     epochs, train_loss, eval_loss,
               os.path.join(args.output_dir, "loss_curve.png"))
    plot_accuracy( epochs, accuracy,
               os.path.join(args.output_dir, "accuracy_curve.png"))
    plot_combined( epochs, train_loss, eval_loss, accuracy,
               os.path.join(args.output_dir, "combined_curves.png"))

    # Print table
    print("\nEpoch  Train Loss  Eval Loss  Eval Accuracy")
    print("-----  ----------  ---------  -------------")
    for e, tl, el, a in zip(epochs, train_loss, eval_loss, accuracy):
        print(f"  {e:3d}  {tl:10.4f}  {el:9.4f}  {a:12.2f}%")


if __name__ == "__main__":
    main()
