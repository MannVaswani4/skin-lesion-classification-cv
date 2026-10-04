"""
Baseline Evaluation & Reporting Script.
Evaluates saved model on the test set, computes all paper metrics + Stage 3 QWK & per-class recalls,
generates confusion matrix plot, per_class_metrics.csv, metrics.json, and BASELINE_REPORT.md.

USAGE:
  python src/evaluate.py --config configs/baseline.yaml --model-path experiments/baseline/best_model.h5
"""
import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

from src.utils import load_config, ensure_dir
from src.augmentation import create_data_generators
from src.dataset import build_flow_generators, CLASS_MAPPING
from src.metrics import compute_all_metrics, CLASS_NAMES
from src.attention import SpatialAttention, ChannelAttention, TemporalAttention

PAPER_REFERENCE_RESULTS = {
    'Accuracy': 92.73,
    'Precision': 92.84,
    'Sensitivity (Recall)': 92.73,
    'F1 Score': 92.70,
    'Overall JAC score': 87.08,
    'Overall DIC Score': 92.70,
    'Overall MCC Score': 91.55,
    'Specificity': 98.79,
    'AUC': 99.42
}

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate baseline model on test set.")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Path to config file.")
    parser.add_argument("--model-path", type=str, default="experiments/baseline/best_model.h5", help="Path to saved model.")
    return parser.parse_args()

def plot_confusion_matrix(cm, class_names, output_path):
    """Plot and save confusion matrix figure."""
    plt.figure(figsize=(9, 7))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Baseline Model Confusion Matrix", fontsize=14, pad=12)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, f"{cm[i, j]}",
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def generate_baseline_report(metrics_dict, per_class_df, cm, config, output_path):
    """Generate human-readable BASELINE_REPORT.md file."""
    report = []
    report.append("# Baseline Reproduction Report")
    report.append("")
    report.append("> **Research Paper**: *An Attention Based Hybrid Approach Using CNN and BiLSTM for Improved Skin Lesion Classification* (Scientific Reports, 2025)")
    report.append("> **Baseline Model**: Custom CNN + Spatial & Channel Attention + Reshape + 2x BiLSTM + Temporal Attention")
    report.append("")
    report.append("---")
    report.append("")
    report.append("## 1. Experimental Configuration & Dataset Summary")
    report.append(f"- **Random Seed**: `{config['training']['seed']}` (Implementation choice)")
    report.append(f"- **Input Shape**: `{config['model']['input_shape']}`")
    report.append(f"- **Batch Size**: `{config['training']['batch_size']}`")
    report.append(f"- **Optimizer**: `{config['training']['optimizer']}` (Initial LR = `{config['training']['initial_lr']}` implementation choice)")
    report.append(f"- **Loss**: `{config['training']['loss']}`")
    report.append(f"- **Dataset Path**: `{config['dataset']['root_dir']}`")
    report.append("- **Dataset Counts**:")
    report.append("  - Train: 32,851 images (4,693 per class)")
    report.append("  - Validation: 4,697 images (671 per class)")
    report.append("  - Test: 9,387 images (1,341 per class)")
    report.append("  - Total: 46,935 images")
    report.append("")
    report.append("### Dataset Leakage Limitation Notice")
    report.append("> **Important**: 2,722 original source lesion IDs present in `train` also have augmented variants in `test`. Per reproduction guidelines, we preserve the dataset split exactly as provided. However, cross-split source-ID overlap means test evaluation reflects model recognition of augmented lesion variants rather than independent-lesion generalization.")
    report.append("")
    report.append("---")
    report.append("")
    report.append("## 2. Benchmark Comparison Against Reported Paper Results")
    report.append("")
    report.append("| Metric | Paper Reported Result | Reproduction Result | Reproduction Gap |")
    report.append("| :--- | :---: | :---: | :---: |")

    for metric, paper_val in PAPER_REFERENCE_RESULTS.items():
        our_val = metrics_dict.get(metric, 0.0)
        gap = our_val - paper_val
        gap_str = f"+{gap:.2f}%" if gap >= 0 else f"{gap:.2f}%"
        report.append(f"| **{metric}** | {paper_val:.2f}% | {our_val:.2f}% | **{gap_str}** |")

    qwk_val = metrics_dict.get('Quadratic Weighted Kappa (QWK)', 0.0)
    report.append(f"| **Quadratic Weighted Kappa (QWK)** | *Not Reported by Paper* | **{qwk_val:.4f}** | N/A (Stage 3 Requirement) |")
    report.append("")
    report.append("*Note: QWK is an assignment Stage 3 evaluation metric and was NOT reported by the paper authors. It was NOT used for training, early stopping, or hyperparameter selection.*")
    report.append("")
    report.append("---")
    report.append("")
    report.append("## 3. Per-Class Recall & Performance Breakdown")
    report.append("")
    report.append(per_class_df.to_markdown(index=False))
    report.append("")
    report.append("---")
    report.append("")
    report.append("## 4. Test Set Confusion Matrix")
    report.append("")
    report.append("```")
    report.append(f"{'Class':<8} " + " ".join([f"{c:>7}" for c in CLASS_NAMES]))
    for i, row in enumerate(cm):
        report.append(f"{CLASS_NAMES[i]:<8} " + " ".join([f"{val:>7}" for val in row]))
    report.append("```")
    report.append("")
    report.append("---")
    report.append("")
    report.append("## 5. Key Findings & Conclusion")
    report.append("- Baseline reproduction executed strictly per paper protocol parameters.")
    report.append("- All results recorded without post-hoc tuning or selective hyperparameter searching.")
    report.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(report))

def main():
    args = parse_args()
    config = load_config(args.config)

    import tensorflow as tf

    custom_objects = {
        'SpatialAttention': SpatialAttention,
        'ChannelAttention': ChannelAttention,
        'TemporalAttention': TemporalAttention
    }

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found at: {args.model_path}")

    print(f"Loading model from: {args.model_path}...")
    model = tf.keras.models.load_model(args.model_path, custom_objects=custom_objects)

    print("Building Data Generators for Evaluation...")
    train_datagen, val_test_datagen = create_data_generators(config)
    _, _, test_gen = build_flow_generators(config, train_datagen, val_test_datagen)

    print("Running predictions on test set...")
    test_gen.reset()
    y_pred_probs = model.predict(test_gen, verbose=1)
    
    y_true_indices = test_gen.classes
    num_classes = config['dataset']['num_classes']
    y_true_onehot = tf.keras.utils.to_categorical(y_true_indices, num_classes=num_classes)

    print("Computing metrics...")
    metrics_dict, cm, per_class_df = compute_all_metrics(y_true_onehot, y_pred_probs)

    print("\n=================== REPRODUCTION TEST RESULTS ===================")
    for metric_name, val in metrics_dict.items():
        if "Kappa" in metric_name:
            print(f"{metric_name:32s}: {val:.4f}")
        else:
            print(f"{metric_name:32s}: {val:.2f}%")
    print("=================================================================\n")

    # Save results
    results_dir = config['outputs']['results_dir']
    ensure_dir(results_dir)

    # 1. Save metrics.json
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics_dict, f, indent=2)

    # 2. Save per_class_metrics.csv
    per_class_df.to_csv(os.path.join(results_dir, "per_class_metrics.csv"), index=False)

    # 3. Save confusion_matrix.png
    plot_confusion_matrix(cm, CLASS_NAMES, os.path.join(results_dir, "confusion_matrix.png"))

    # 4. Save BASELINE_REPORT.md
    generate_baseline_report(
        metrics_dict, per_class_df, cm, config,
        os.path.join(results_dir, "BASELINE_REPORT.md")
    )

    print(f"Evaluation complete! All artifacts saved to: {results_dir}")

if __name__ == "__main__":
    main()
