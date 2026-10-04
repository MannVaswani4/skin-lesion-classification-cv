"""
Preprocessing module for runtime image transformation and normalization.
Resizes 128x128 source images to 224x224 and normalizes pixel values by 1/255.
"""

def get_target_size(config):
    """Return target image size tuple (height, width)."""
    size = config['dataset']['target_image_size']
    return tuple(size)

def get_rescale_factor(config):
    """Return pixel rescale factor."""
    return config['preprocessing']['rescale_factor']
