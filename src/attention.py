"""
Custom Attention Layers implemented per paper specifications:
- SpatialAttention: Channel pooling -> Conv2D(7x7) -> Sigmoid -> Element-wise multiply
- ChannelAttention: Global Pooling (Avg & Max) -> Shared MLP -> Sigmoid -> Element-wise multiply
- TemporalAttention: Hidden projection (tanh) -> Context dot product -> Softmax -> Weighted sum over time
"""

try:
    import tensorflow as tf
    from tensorflow.keras import layers
except ImportError:
    # Graceful fallback for environments before TF installation
    tf = None
    layers = object

if tf is not None:

    class SpatialAttention(layers.Layer):
        """
        Spatial Attention Module.
        Computes channel-wise average and max pooling, concatenates them,
        applies 7x7 Conv2D with Sigmoid activation, and multiplies with input.
        """
        def __init__(self, kernel_size=7, **kwargs):
            super(SpatialAttention, self).__init__(**kwargs)
            self.kernel_size = kernel_size
            self.conv = layers.Conv2D(
                filters=1,
                kernel_size=kernel_size,
                padding='same',
                activation='sigmoid',
                use_bias=False
            )

        def call(self, inputs):
            avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
            max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
            concat = tf.concat([avg_pool, max_pool], axis=-1)
            attention_map = self.conv(concat)
            return inputs * attention_map

        def get_config(self):
            config = super(SpatialAttention, self).get_config()
            config.update({'kernel_size': self.kernel_size})
            return config

    class ChannelAttention(layers.Layer):
        """
        Channel Attention Module.
        Applies Global Average Pooling and Global Max Pooling,
        processes through a shared MLP, combines with Sigmoid, and multiplies with input.
        """
        def __init__(self, reduction_ratio=16, **kwargs):
            super(ChannelAttention, self).__init__(**kwargs)
            self.reduction_ratio = reduction_ratio

        def build(self, input_shape):
            channels = input_shape[-1]
            reduced_channels = max(1, channels // self.reduction_ratio)
            self.shared_mlp_dense1 = layers.Dense(
                reduced_channels, activation='relu', use_bias=True
            )
            self.shared_mlp_dense2 = layers.Dense(
                channels, use_bias=True
            )
            super(ChannelAttention, self).build(input_shape)

        def call(self, inputs):
            avg_pool = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
            max_pool = tf.reduce_max(inputs, axis=[1, 2], keepdims=True)

            avg_out = self.shared_mlp_dense2(self.shared_mlp_dense1(avg_pool))
            max_out = self.shared_mlp_dense2(self.shared_mlp_dense1(max_pool))

            attention_map = tf.nn.sigmoid(avg_out + max_out)
            return inputs * attention_map

        def get_config(self):
            config = super(ChannelAttention, self).get_config()
            config.update({'reduction_ratio': self.reduction_ratio})
            return config

    class TemporalAttention(layers.Layer):
        """
        Temporal Attention Module.
        Projects sequence input X into hidden representation via tanh,
        computes attention score using a context vector, applies softmax over time,
        and computes a weighted sum over time steps.
        """
        def __init__(self, units=128, **kwargs):
            super(TemporalAttention, self).__init__(**kwargs)
            self.units = units

        def build(self, input_shape):
            feature_dim = input_shape[-1]
            self.W = self.add_weight(
                name='W_temp_att',
                shape=(feature_dim, self.units),
                initializer='glorot_uniform',
                trainable=True
            )
            self.b = self.add_weight(
                name='b_temp_att',
                shape=(self.units,),
                initializer='zeros',
                trainable=True
            )
            self.u = self.add_weight(
                name='u_temp_att',
                shape=(self.units, 1),
                initializer='glorot_uniform',
                trainable=True
            )
            super(TemporalAttention, self).build(input_shape)

        def call(self, inputs):
            # inputs shape: (batch_size, time_steps, feature_dim)
            uit = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
            # uit shape: (batch_size, time_steps, units)
            ait = tf.tensordot(uit, self.u, axes=1)  # (batch_size, time_steps, 1)
            ait = tf.nn.softmax(ait, axis=1)
            # Weighted sum over time steps
            attended_representation = tf.reduce_sum(inputs * ait, axis=1)
            return attended_representation

        def get_config(self):
            config = super(TemporalAttention, self).get_config()
            config.update({'units': self.units})
            return config
else:
    SpatialAttention = None
    ChannelAttention = None
    TemporalAttention = None
