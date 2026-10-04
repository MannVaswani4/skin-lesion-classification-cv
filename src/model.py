"""
Proposed Model Builder Module.
Constructs the hybrid CNN + BiLSTM + Attention model strictly according to paper specifications.
"""

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from src.attention import SpatialAttention, ChannelAttention, TemporalAttention
except ImportError:
    tf = None

def build_baseline_model(config):
    """
    Build and return the CNN + BiLSTM + Attention baseline model.
    """
    if tf is None:
        raise ImportError("TensorFlow is required to build the model.")

    input_shape = tuple(config['model']['input_shape'])
    num_classes = config['model']['num_classes']

    inputs = layers.Input(shape=input_shape, name="input_image")

    # CNN Feature Extractor
    # Block 1
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name="conv1")(inputs)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name="conv2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name="conv3")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)

    # Block 4
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name="conv4")(x)
    x = layers.MaxPooling2D((2, 2), name="pool4")(x)
    # Feature map shape: (batch, 14, 14, 256)

    # Spatial Attention
    x = SpatialAttention(kernel_size=7, name="spatial_attention")(x)

    # Channel Attention
    r = config['model'].get('channel_attention_reduction_ratio', 16)
    x = ChannelAttention(reduction_ratio=r, name="channel_attention")(x)

    # Reshape 2D spatial feature map into 1D sequence format for BiLSTM
    # Shape transition: (14, 14, 256) -> (196, 256)
    x = layers.Reshape((196, 256), name="reshape_to_sequence")(x)

    # BiLSTM Layer 1
    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True),
        name="bilstm_1"
    )(x)

    # BiLSTM Layer 2
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True),
        name="bilstm_2"
    )(x)

    # Temporal Attention
    x = TemporalAttention(units=128, name="temporal_attention")(x)

    # Classification Head
    x = layers.Dense(512, activation='relu', name="dense_512")(x)
    x = layers.Dropout(0.5, name="dropout_0.5")(x)
    outputs = layers.Dense(num_classes, activation='softmax', name="classifier_output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="CNN_BiLSTM_Attention")
    return model
