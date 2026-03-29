import argparse
import os
import sys
from typing import Optional

_cfg = os.path.dirname(os.path.abspath(__file__))
if _cfg not in sys.path:
    sys.path.insert(0, _cfg)
from pipeline_data import tunnel_output_dir, ABLATION_CONDITIONS, ABLATION_CODES

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, jaccard_score

# Semantic segment count: max class id in GT/pred (Background=0, …).
SEGMENT_SCHEMAS = {
    6: {
        "max_class_id": 6,
        "label": "6-class (B2 at index 6)",
        "class_names": {
            0: "Background",
            1: "K-block",
            2: "B1-block",
            3: "A1-block",
            4: "A2-block",
            5: "A3-block",
            6: "B2-block",
        },
    },
    7: {
        "max_class_id": 7,
        "label": "7-class (A4 at 6, B2 at 7)",
        "class_names": {
            0: "Background",
            1: "K-block",
            2: "B1-block",
            3: "A1-block",
            4: "A2-block",
            5: "A3-block",
            6: "A4-block",
            7: "B2-block",
        },
    },
}


def evaluation_output_dir(input_dir: str) -> str:
    """All semantic metrics live under ``evaluation/``; schema (6 vs 7) is in ``performance*.md``."""
    return os.path.join(input_dir, "evaluation")


def infer_segment_schema(gt_labels: np.ndarray, pred_labels: np.ndarray) -> int:
    """
    Choose 6- vs 7-class naming from **ground truth only**.

    Predictions may include 7, 8, or NaN (unmapped); using max(pred) wrongly forces
    7-class for 6-class tunnels (e.g. 1-4: GT max 6, pred has 8).
    """
    del pred_labels  # API unchanged; schema follows GT convention
    gt = np.asarray(gt_labels, dtype=np.float64)
    mx_gt = float(np.nanmax(gt)) if np.any(np.isfinite(gt)) else 0.0
    return 7 if mx_gt > 6 else 6


def parse_args():
    p = argparse.ArgumentParser(
        description="Segmentation evaluation from only_label.csv (6- or 7-class schema)."
    )
    p.add_argument("tunnel_id", help="Tunnel id, e.g. 4-1")
    p.add_argument(
        "--ablation", "-a",
        required=True,
        choices=ABLATION_CODES,
        help=f"Ablation condition code: {', '.join(ABLATION_CODES)}",
    )
    p.add_argument(
        "--schema",
        choices=("auto", "6", "7", "both"),
        default="auto",
        help="auto: GT max label >6 => 7-class; 6/7: force schema; both: write evaluation/ with performance_6.md and performance_7.md",
    )
    return p.parse_args()


def calculate_metrics(gt_labels, pred_labels, class_names=None):
    """
    Calculate segmentation metrics:
    - Overall Accuracy (OA)
    - F1 Score
    - Mean IoU (mIoU)
    - Per-class IoU
    """
    gt_labels = np.array(gt_labels)
    pred_labels = np.array(pred_labels)

    if gt_labels.ndim > 1:
        gt_flat = gt_labels.flatten()
        pred_flat = pred_labels.flatten()
    else:
        gt_flat = gt_labels
        pred_flat = pred_labels

    classes = np.unique(np.concatenate((np.unique(gt_flat), np.unique(pred_flat))))
    classes = np.sort(classes)

    oa = accuracy_score(gt_flat, pred_flat)
    f1 = f1_score(gt_flat, pred_flat, average="macro", labels=classes, zero_division=0)
    iou_per_class = jaccard_score(
        gt_flat, pred_flat, average=None, labels=classes, zero_division=0
    )
    miou = np.mean(iou_per_class)

    results = {
        "OA": oa,
        "F1": f1,
        "mIoU": miou,
        "IoU_per_class": iou_per_class,
        "classes": classes,
    }

    print(f"OA {oa:.3f} F1 {f1:.3f} mIoU {miou:.3f}")

    if class_names is None:
        class_names = {c: f"Class {c}" for c in classes}

    print("Per-class IoU:", end=" ")
    class_iou_strs = []
    for i, class_idx in enumerate(classes):
        class_name = class_names.get(class_idx, f"Class {class_idx}")
        class_iou_strs.append(f"{class_name} {iou_per_class[i]:.3f}")

    print(", ".join(class_iou_strs))

    return results


def plot_confusion_matrices(
    gt_labels,
    pred_labels,
    class_names,
    present_classes=None,
    output_file="confusion_matrices.png",
):
    gt_labels = np.array(gt_labels)
    pred_labels = np.array(pred_labels)

    if gt_labels.ndim > 1:
        gt_flat = gt_labels.flatten()
        pred_flat = pred_labels.flatten()
    else:
        gt_flat = gt_labels
        pred_flat = pred_labels

    if present_classes is None:
        present_classes = sorted(
            list(set(np.unique(gt_flat)) | set(np.unique(pred_flat)))
        )

    class_labels = [class_names.get(c, f"Class {c}") for c in present_classes]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    cm = confusion_matrix(gt_flat, pred_flat, labels=present_classes)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
        ax=ax1,
    )
    ax1.set_title("Confusion Matrix (Raw Counts)")
    ax1.set_ylabel("Ground Truth")
    ax1.set_xlabel("Prediction")

    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
        ax=ax2,
    )
    ax2.set_title("Confusion Matrix (Normalized by Row)")
    ax2.set_ylabel("Ground Truth")
    ax2.set_xlabel("Prediction")

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

    print(f"\nConfusion matrices saved to '{output_file}'")


def plot_iou_bars(iou_per_class, classes, class_names, output_file="iou_by_class.png"):
    class_labels = [class_names.get(c, f"Class {c}") for c in classes]

    plt.figure(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(classes)))

    bars = plt.bar(class_labels, iou_per_class, color=colors)
    plt.axhline(
        y=np.mean(iou_per_class),
        color="r",
        linestyle="-",
        label=f"Mean IoU: {np.mean(iou_per_class):.3f}",
    )

    for bar, iou in zip(bars, iou_per_class):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{iou:.3f}",
            ha="center",
            va="bottom",
            rotation=0,
        )

    plt.xlabel("Class")
    plt.ylabel("IoU Score")
    plt.title("IoU Scores by Class")
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

    print(f"IoU by class chart saved to '{output_file}'")


def class_distribution_plot(
    gt_labels, pred_labels, class_names, output_file="class_distribution.png"
):
    gt_labels = np.array(gt_labels)
    pred_labels = np.array(pred_labels)

    if gt_labels.ndim > 1:
        gt_flat = gt_labels.flatten()
        pred_flat = pred_labels.flatten()
    else:
        gt_flat = gt_labels
        pred_flat = pred_labels

    classes = sorted(list(set(np.unique(gt_flat)) | set(np.unique(pred_flat))))

    gt_counts = np.array([np.sum(gt_flat == c) for c in classes])
    pred_counts = np.array([np.sum(pred_flat == c) for c in classes])

    class_labels = [class_names.get(c, f"Class {c}") for c in classes]

    plt.figure(figsize=(10, 8))
    x = np.arange(len(classes))
    width = 0.35

    plt.bar(x - width / 2, gt_counts, width, label="Ground Truth")
    plt.bar(x + width / 2, pred_counts, width, label="Prediction")

    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("Class Distribution (Counts)")
    plt.xticks(x, class_labels)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    if np.max(gt_counts) > 1000 or np.max(pred_counts) > 1000:
        from matplotlib.ticker import FuncFormatter

        def format_func(x, pos):
            if x >= 1e6:
                return f"{x * 1e-6:.1f}M"
            if x >= 1e3:
                return f"{x * 1e-3:.1f}K"
            return f"{x:.0f}"

        plt.gca().yaxis.set_major_formatter(FuncFormatter(format_func))

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

    print(f"Class distribution (counts) plot saved to '{output_file}'")


def visualize_results(
    gt_labels,
    pred_labels,
    class_names=None,
    cmap="tab10",
    output_file="segmentation_comparison.png",
):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(gt_labels, cmap=cmap)
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(pred_labels, cmap=cmap)
    plt.title("Prediction")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

    print(f"Segmentation comparison saved to '{output_file}'")


def generate_example_data(class_names):
    np.random.seed(42)

    h, w = 500, 500
    gt_labels = np.zeros((h, w), dtype=np.int32)

    num_classes = len(class_names) if isinstance(class_names, dict) else len(class_names)

    for i in range(num_classes):
        mask = np.random.rand(h, w) < 0.15
        gt_labels[mask] = i

    pred_labels = gt_labels.copy()
    noise_mask = np.random.rand(h, w) < 0.12
    pred_labels[noise_mask] = np.random.randint(0, num_classes, size=np.sum(noise_mask))

    return gt_labels, pred_labels


def evaluate_csv_data(
    tunnel_id: str,
    input_dir: str,
    segment_schema: Optional[int],
    artifact_suffix: str = "",
):
    """
    Evaluate using data/{tunnel_id}/only_label.csv.

    segment_schema: 6, 7, or None (infer from GT).
    artifact_suffix: e.g. \"_6\" / \"_7\" when --schema both (same evaluation/ dir).
    """
    data_path = os.path.join(input_dir, "only_label.csv")
    try:
        df = pd.read_csv(data_path)
        print(f"Successfully loaded {data_path}")
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None

    gt_labels = df["gt_labels"].values
    pred_labels = df["pred_labels"].values

    if segment_schema is None:
        segment_schema = infer_segment_schema(gt_labels, pred_labels)
        print(
            f"Auto-selected schema: {segment_schema} "
            f"({SEGMENT_SCHEMAS[segment_schema]['label']})"
        )

    cfg = SEGMENT_SCHEMAS[segment_schema]
    max_id = cfg["max_class_id"]
    class_names = cfg["class_names"]

    original_size = len(gt_labels)
    valid_mask = (
        np.isfinite(gt_labels)
        & np.isfinite(pred_labels)
        & (gt_labels <= max_id)
        & (pred_labels <= max_id)
    )

    gt_beyond = gt_labels[gt_labels > max_id]
    pred_beyond = pred_labels[pred_labels > max_id]
    if len(gt_beyond) > 0 or len(pred_beyond) > 0:
        print(f"Filtering out classes beyond {max_id}:")
        for cls in np.unique(np.concatenate([gt_beyond, pred_beyond])):
            gt_count = np.sum(gt_labels == cls)
            pred_count = np.sum(pred_labels == cls)
            print(f"  Class {cls}: GT={gt_count}, Pred={pred_count} points")

    gt_labels = gt_labels[valid_mask]
    pred_labels = pred_labels[valid_mask]

    filtered_size = len(gt_labels)
    if original_size != filtered_size:
        pct = 100 * (original_size - filtered_size) / original_size
        print(f"Filtered dataset: {original_size} -> {filtered_size} points ({pct:.2f}% removed)")

    unique_gt = np.unique(gt_labels)
    unique_pred = np.unique(pred_labels)
    print(f"Unique ground truth labels: {unique_gt}")
    print(f"Unique predicted labels: {unique_pred}")

    print(f"\n--- Evaluation ({segment_schema}-class) for {tunnel_id} ---")
    results = calculate_metrics(gt_labels, pred_labels, class_names)

    print("\nClass distribution:")
    print("Ground truth:")
    for class_idx, name in class_names.items():
        count = np.sum(gt_labels == class_idx)
        percentage = 100 * count / len(gt_labels)
        if count > 0:
            print(f"  {name}: {count} ({percentage:.2f}%)")

    print("\nPredictions:")
    for class_idx, name in class_names.items():
        count = np.sum(pred_labels == class_idx)
        percentage = 100 * count / len(pred_labels)
        if count > 0:
            print(f"  {name}: {count} ({percentage:.2f}%)")

    output_dir = evaluation_output_dir(input_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("\nGenerating visualizations...")
    plot_iou_bars(
        results["IoU_per_class"],
        results["classes"],
        class_names,
        os.path.join(output_dir, f"iou_by_class{artifact_suffix}.png"),
    )
    class_distribution_plot(
        gt_labels,
        pred_labels,
        class_names,
        os.path.join(output_dir, f"class_distribution{artifact_suffix}.png"),
    )

    markdown_path = os.path.join(output_dir, f"performance{artifact_suffix}.md")
    with open(markdown_path, "w") as f:
        f.write(f"# Performance Metrics for Tunnel {tunnel_id}\n\n")
        f.write(f"- Segment schema: **{segment_schema}-class** ({cfg['label']})\n\n")
        f.write("## Overall Metrics\n")
        f.write(f"- Overall Accuracy (OA): {results['OA']:.3f}\n")
        f.write(f"- F1 Score: {results['F1']:.3f}\n")
        f.write(f"- Mean IoU (mIoU): {results['mIoU']:.3f}\n\n")

        f.write("## Per-class IoU\n")
        for i, class_idx in enumerate(results["classes"]):
            class_name = class_names.get(class_idx, f"Class {class_idx}")
            f.write(f"- {class_name}: {results['IoU_per_class'][i]:.3f}\n")

    print(f"Performance metrics saved to {markdown_path}")

    print(f"\nResults for {tunnel_id} ({segment_schema}-class):")
    print(f"OA {results['OA']:.3f} F1 {results['F1']:.3f} mIoU {results['mIoU']:.3f}")

    return results


def evaluate_instance_segmentation(tunnel_id: str, input_dir: str):
    data_path = os.path.join(input_dir, "only_label.csv")
    try:
        df = pd.read_csv(data_path)
        if "gt_rings" not in df.columns or "pred_rings" not in df.columns:
            print("The CSV file does not contain ring data columns (gt_rings, pred_rings)")
            return

        print(f"\n--- Instance Segmentation Evaluation for {tunnel_id} ---")
        print("Evaluating ring-based instance segmentation")

        gt_rings = df["gt_rings"].values
        pred_rings = df["pred_rings"].values

        unique_gt_rings = np.unique(gt_rings)
        unique_pred_rings = np.unique(pred_rings)
        print(f"Unique ground truth rings: {unique_gt_rings}")
        print(f"Unique predicted rings: {unique_pred_rings}")

        ring_accuracy = np.mean(gt_rings == pred_rings)
        print(f"Ring prediction accuracy: {ring_accuracy:.3f}")

    except Exception as e:
        print(f"Error in instance segmentation evaluation: {e}")


def main():
    args = parse_args()
    tunnel_id = args.tunnel_id

    cond = ABLATION_CONDITIONS[args.ablation]
    os.environ["R4TUN_PIPELINE_OUT_PREFIX"] = cond["out_prefix"]

    input_dir = tunnel_output_dir(tunnel_id).rstrip("/")

    print(f"Starting evaluation for tunnel: {tunnel_id}")
    print(f"=== Segmentation Evaluation Tool for Tunnel {tunnel_id} ===")

    if args.schema == "both":
        print("\nEvaluating semantic segmentation (6-class and 7-class)...")
        evaluate_csv_data(tunnel_id, input_dir, 6, "_6")
        evaluate_csv_data(tunnel_id, input_dir, 7, "_7")
    elif args.schema == "auto":
        print("\nEvaluating semantic segmentation (schema auto)...")
        evaluate_csv_data(tunnel_id, input_dir, None)
    else:
        print(f"\nEvaluating semantic segmentation (--schema {args.schema})...")
        evaluate_csv_data(tunnel_id, input_dir, int(args.schema))

    print(f"\n=== Evaluation Complete for Tunnel {tunnel_id} ===")


if __name__ == "__main__":
    main()
