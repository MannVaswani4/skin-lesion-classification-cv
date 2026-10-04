"""
Augmentation module implementing the exact data augmentation pipeline
specified in Table 4 of the paper using Keras ImageDataGenerator parameters.
"""

def create_data_generators(config):
    """
    Create ImageDataGenerator instances for training, validation, and testing.
    
    Training ImageDataGenerator includes:
    - rescale: 1/255
    - rotation_range: 20
    - width_shift_range: 0.1
    - height_shift_range: 0.1
    - zoom_range: 0.1
    - shear_range: 10
    - horizontal_flip: True
    - fill_mode: "nearest"
    
    Validation/Testing ImageDataGenerator includes:
    - rescale: 1/255
    """
    try:
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
    except ImportError:
        raise ImportError("TensorFlow/Keras is required to create ImageDataGenerators.")

    aug_cfg = config['augmentation']
    rescale = config['preprocessing']['rescale_factor']

    train_datagen = ImageDataGenerator(
        rescale=rescale,
        rotation_range=aug_cfg['rotation_range'],
        width_shift_range=aug_cfg['width_shift_range'],
        height_shift_range=aug_cfg['height_shift_range'],
        zoom_range=aug_cfg['zoom_range'],
        shear_range=aug_cfg['shear_range'],
        horizontal_flip=aug_cfg['horizontal_flip'],
        fill_mode=aug_cfg['fill_mode']
    )

    val_test_datagen = ImageDataGenerator(
        rescale=rescale
    )

    return train_datagen, val_test_datagen
