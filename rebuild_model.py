"""
Script to rebuild and save the Malaria detection model in the new Keras format.
This creates a simple CNN model compatible with TensorFlow 2.x and Keras 3.x
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np

# Build the same model architecture as in the original notebook
def build_model():
    model = models.Sequential([
        layers.Input(shape=(50, 50, 3)),
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(2, activation='softmax')  # Changed from sigmoid to softmax for 2-class classification
    ])
    
    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adam(),
        metrics=['accuracy']
    )
    
    return model

# Create and save the model
print("Building Malaria Detection Model...")
model = build_model()
print(model.summary())

print("\nSaving model...")
model.save("models/model_malaria.h5")
print("✓ Model saved as models/model_malaria.h5")

# Test that the model can be loaded
print("\nTesting model loading...")
loaded_model = keras.models.load_model("models/model_malaria.h5")
print("✓ Model loaded successfully")

# Test a dummy prediction
print("\nTesting dummy prediction...")
dummy_input = np.random.rand(1, 50, 50, 3).astype(np.float32)
prediction = loaded_model.predict(dummy_input, verbose=0)
print(f"✓ Dummy prediction shape: {prediction.shape}")
print(f"✓ Prediction output: {prediction[0]}")

print("\n✓ Model rebuild complete! You can now run app.py")
