"""
Dataset loading module.
Discovers file paths and builds data generators for train, validation, and test splits
without altering the original dataset filesystem layout or files.
"""
import os
import glob

CLASS_MAPPING = {
    "AKIEC": 0,
    "BCC": 1,
    "BKL": 2,
    "DF": 3,
    "MEL": 4,
    "NV": 5,
    "VASC": 6
}

def verify_dataset_structure(data_root):
    """
    Verify presence of train, validation, and test directories and return counts per split.
    """
    splits = ['train', 'validation', 'test']
    counts = {}
    for split in splits:
        split_dir = os.path.join(data_root, split)
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Dataset split directory missing: {split_dir}")
        split_counts = {}
        for cls in CLASS_MAPPING.keys():
            cls_dir = os.path.join(split_dir, cls)
            if not os.path.exists(cls_dir):
                raise FileNotFoundError(f"Class directory missing: {cls_dir}")
            files = glob.glob(os.path.join(cls_dir, "*.jpg"))
            split_counts[cls] = len(files)
        counts[split] = split_counts
    return counts

def build_flow_generators(config, train_datagen, val_test_datagen):
    """
    Build Keras DirectoryIterator generators for train, validation, and test splits.
    """
    root_dir = config['dataset']['root_dir']
    target_size = tuple(config['dataset']['target_image_size'])
    batch_size = config['training']['batch_size']
    seed = config['training']['seed']

    train_dir = os.path.join(root_dir, 'train')
    val_dir = os.path.join(root_dir, 'validation')
    test_dir = os.path.join(root_dir, 'test')

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        seed=seed
    )

    val_generator = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    test_generator = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, val_generator, test_generator
