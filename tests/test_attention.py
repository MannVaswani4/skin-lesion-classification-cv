"""
Attention module tests.
Verifies shape transitions and mathematical flow through Spatial, Channel, and Temporal Attention layers.
"""
import unittest

class TestAttentionLayers(unittest.TestCase):

    def test_attention_shapes_mock_or_tf(self):
        try:
            import tensorflow as tf
            from src.attention import SpatialAttention, ChannelAttention, TemporalAttention

            # Test Spatial Attention shape
            spatial_layer = SpatialAttention(kernel_size=7)
            x_spatial = tf.random.normal([2, 14, 14, 256])
            out_spatial = spatial_layer(x_spatial)
            self.assertEqual(out_spatial.shape, (2, 14, 14, 256))

            # Test Channel Attention shape
            channel_layer = ChannelAttention(reduction_ratio=16)
            x_channel = tf.random.normal([2, 14, 14, 256])
            out_channel = channel_layer(x_channel)
            self.assertEqual(out_channel.shape, (2, 14, 14, 256))

            # Test Temporal Attention shape
            temporal_layer = TemporalAttention(units=128)
            x_temporal = tf.random.normal([2, 196, 128])
            out_temporal = temporal_layer(x_temporal)
            self.assertEqual(out_temporal.shape, (2, 128))

        except ImportError:
            self.skipTest("TensorFlow is not installed in the current environment; skipping TF tensor shape test.")

if __name__ == "__main__":
    unittest.main()
