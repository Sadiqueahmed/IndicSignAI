#!/usr/bin/env python3
"""
Convert ISL_IMAGE.keras model to TFLite for faster real-time inference.
Usage: python convert_to_tflite.py

The TFLite model runs 3-10x faster on CPU than the full Keras model.
"""

import os
import sys
import time
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
os.chdir(os.path.join(os.path.dirname(__file__), 'src'))

import tensorflow as tf
from tensorflow import keras
from models.isl_image_model import TransformerBlock, ISLImageModel


def convert_to_tflite():
    print("=" * 60)
    print("ISL_IMAGE.keras -> TFLite Conversion")
    print("=" * 60)

    # Load the model
    model_handler = ISLImageModel()
    if not model_handler.load_model():
        print("[X] Failed to load model")
        return False

    model = model_handler.model
    print(f"[OK] Model loaded: {model.input_shape} -> {model.output_shape}")

    # Create a concrete function for TFLite conversion
    # This handles the custom TransformerBlock by tracing the graph
    @tf.function(input_signature=[tf.TensorSpec(shape=[1, 160, 160, 3], dtype=tf.float32)])
    def predict(x):
        return model(x, training=False)

    concrete_func = predict.get_concrete_function()

    # Convert with float16 quantization (good balance of speed and accuracy)
    print("\nConverting to TFLite with float16 quantization...")
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    try:
        tflite_model = converter.convert()
    except Exception as e:
        print(f"[!] Float16 quantization failed: {e}")
        print("    Trying without quantization...")
        converter2 = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        tflite_model = converter2.convert()

    # Save the TFLite model
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ISL_IMAGE.tflite')
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    # Size comparison
    model_dir = model_handler.model_path
    if os.path.isdir(model_dir):
        original_size = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, dn, filenames in os.walk(model_dir)
            for f in filenames
        )
    else:
        original_size = os.path.getsize(model_dir)

    tflite_size = os.path.getsize(output_path)
    print(f"\n[OK] TFLite model saved: {output_path}")
    print(f"     Original: {original_size / 1024 / 1024:.1f} MB")
    print(f"     TFLite:   {tflite_size / 1024 / 1024:.1f} MB")
    print(f"     Reduction: {(1 - tflite_size / max(original_size, 1)) * 100:.0f}%")

    # Verify the TFLite model
    print("\nVerifying TFLite model...")
    interpreter = tf.lite.Interpreter(model_path=output_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    test_input = np.random.randn(1, 160, 160, 3).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    print(f"[OK] Verification passed - output shape: {output.shape}")

    # Benchmark: Keras vs TFLite
    N = 20
    print(f"\nBenchmarking ({N} inferences)...")

    # Keras benchmark
    start = time.time()
    for _ in range(N):
        model.predict(test_input, verbose=0)
    keras_time = (time.time() - start) / N * 1000

    # Direct call benchmark
    start = time.time()
    for _ in range(N):
        model(test_input, training=False)
    direct_time = (time.time() - start) / N * 1000

    # TFLite benchmark
    start = time.time()
    for _ in range(N):
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
    tflite_time = (time.time() - start) / N * 1000

    print(f"  Keras .predict():   {keras_time:.1f} ms/inference")
    print(f"  Keras direct call:  {direct_time:.1f} ms/inference")
    print(f"  TFLite:             {tflite_time:.1f} ms/inference")
    print(f"  TFLite speedup:     {keras_time / max(tflite_time, 0.1):.1f}x vs predict()")
    print(f"                      {direct_time / max(tflite_time, 0.1):.1f}x vs direct call")

    print("\n" + "=" * 60)
    print("[OK] Conversion complete! Restart the server to use TFLite.")
    print("=" * 60)
    return True


if __name__ == '__main__':
    convert_to_tflite()
