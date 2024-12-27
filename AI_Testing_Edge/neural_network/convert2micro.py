import tensorflow as tf
import keras
from pathlib import Path

original_model = keras.models.load_model(Path(__file__).parent / "model.keras")

converter = tf.lite.TFLiteConverter.from_keras_model(original_model)
# Kwantyzacja (opcjonalna, dla mniejszych modeli)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Zapisanie modelu jako .tflite
with open(Path(__file__).parent / "model.tflite", "wb") as f:
    f.write(tflite_model)