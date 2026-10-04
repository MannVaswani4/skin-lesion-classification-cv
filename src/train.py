"""
Baseline Training Script.
Sets up seeds, dataset generators, callbacks (ReduceLROnPlateau & EarlyStopping),
compiles model with Adam and categorical_crossentropy, and trains the model.

USAGE:
  python src/train.py --config configs/baseline.yaml [--dry-run]
"""
import os
import argparse
import json
import yaml

from src.utils import set_seeds, load_config, ensure_dir
from src.preprocessing import get_target_size
from src.augmentation import create_data_generators
from src.dataset import build_flow_generators, verify_dataset_structure
from src.model import build_baseline_model

def parse_args():
    parser = argparse.ArgumentParser(description="Train baseline CNN + BiLSTM + Attention model.")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Path to config YAML file.")
    parser.add_argument("--dry-run", action="store_true", help="Build model and compile without executing training.")
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config(args.config)

    # 1. Set seeds
    seed = config['training']['seed']
    set_seeds(seed)

    # 2. Verify dataset
    print("Verifying dataset structure...")
    counts = verify_dataset_structure(config['dataset']['root_dir'])
    print("Dataset counts verified:", json.dumps(counts, indent=2))

    # 3. Create outputs directories
    exp_dir = config['outputs']['experiment_dir']
    results_dir = config['outputs']['results_dir']
    ensure_dir(exp_dir)
    ensure_dir(results_dir)

    # 4. Build Data Generators
    print("Building Data Generators...")
    train_datagen, val_test_datagen = create_data_generators(config)
    train_gen, val_gen, test_gen = build_flow_generators(config, train_datagen, val_test_datagen)

    # 5. Build and Compile Model
    print("Building Baseline Model...")
    model = build_baseline_model(config)
    model.summary()

    import tensorflow as tf
    optimizer = tf.keras.optimizers.Adam(learning_rate=config['training']['initial_lr'])
    model.compile(
        optimizer=optimizer,
        loss=config['training']['loss'],
        metrics=['accuracy']
    )

    # 6. Callbacks
    cb_cfg = config['training']['callbacks']
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=cb_cfg['reduce_lr']['monitor'],
        factor=cb_cfg['reduce_lr']['factor'],
        patience=cb_cfg['reduce_lr']['patience'],
        min_lr=cb_cfg['reduce_lr']['min_lr'],
        verbose=1
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor=cb_cfg['early_stopping']['monitor'],
        patience=cb_cfg['early_stopping']['patience'],
        restore_best_weights=cb_cfg['early_stopping']['restore_best_weights'],
        verbose=1
    )

    checkpoint_path = os.path.join(exp_dir, "best_model.h5")
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )

    callbacks = [reduce_lr, early_stopping, model_checkpoint]

    if args.dry-run:
        print("[DRY-RUN] Model built and compiled successfully. Training skipped per --dry-run flag.")
        return

    print("Starting Model Training...")
    history = model.fit(
        train_gen,
        epochs=config['training']['epochs'],
        validation_data=val_gen,
        callbacks=callbacks
    )

    # Save final model & history log
    model.save(os.path.join(exp_dir, "final_model.h5"))
    with open(os.path.join(exp_dir, "history.json"), "w") as f:
        json.dump(history.history, f, indent=2)

    print("Training complete! Model and logs saved to:", exp_dir)

if __name__ == "__main__":
    main()
