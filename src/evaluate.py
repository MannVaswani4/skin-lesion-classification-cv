"""
Baseline Evaluation Script.
Evaluates saved model on the test set, computes all 9 metrics, generates confusion matrix
and ROC curve plots, and outputs comparison against reported paper results.

USAGE:
  python src/evaluate.py --config configs/baseline.yaml --model-path experiments/baseline/best_model.h5
"""
import os
import argparse
import json
import numpy as np
import yaml

from src.utils import load_config, ensure_dir
from src.augmentation import create_data_generators
from src.dataset import build_flow_generators
from src.metrics import compute_all_metrics
from src.attention import SpatialAttention, ChannelAttention, TemporalAttention

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate baseline model on test set.")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Path to config file.")
    parser.add_argument("--model-path", type=str, default="experiments/baseline/best_model.h5", help="Path to saved model.")
    return parser.parse_args()

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
    
    # Get true one-hot labels
    y_true_indices = test_gen.classes
    num_classes = config['dataset']['num_classes']
    y_true_onehot = tf.keras.utils.to_categorical(y_true_indices, num_classes=num_classes)

    print("Computing metrics...")
    metrics_dict, cm = compute_all_metrics(y_true_onehot, y_pred_probs)

    print("\n=================== REPRODUCTION TEST RESULTS ===================")
    for metric_name, val in metrics_dict.items():
        print(f"{metric_name:25s}: {val:.2f}%")
    print("=================================================================\n")

    # Save results
    results_dir = config['outputs']['results_dir']
    ensure_dir(results_dir)
    with open(os.path.join(results_dir, "test_metrics.json"), "w") as f:
        json.dump(metrics_dict, f, indent=2)

    np.save(os.path.join(results_dir, "confusion_matrix.npy"), cm)
    print(f"Results and metrics saved to: {results_dir}")

if __name__ == "__main__":
    main()
