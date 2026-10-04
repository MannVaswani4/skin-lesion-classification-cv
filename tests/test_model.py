"""
Model architecture unit tests.
Verifies model creation, layer connections, input shape (224, 224, 3), and output shape (7).
"""
import unittest
from src.utils import load_config

class TestModelArchitecture(unittest.TestCase):

    def test_model_build(self):
        try:
            import tensorflow as tf
            from src.model import build_baseline_model

            config = load_config("configs/baseline.yaml")
            model = build_baseline_model(config)

            self.assertEqual(model.input_shape, (None, 224, 224, 3))
            self.assertEqual(model.output_shape, (None, 7))

            # Test dummy forward pass
            dummy_input = tf.random.normal([2, 224, 224, 3])
            output = model(dummy_input)
            self.assertEqual(output.shape, (2, 7))

        except ImportError:
            self.skipTest("TensorFlow is not installed in the current environment; skipping model build test.")

if __name__ == "__main__":
    unittest.main()
