# example.py

from classifiers import HybridCNNViT
import numpy as np

# Create model
model = HybridCNNViT()

# Dummy input (batch of 2 RGB images of 256x256)
x = np.random.rand(2, 256, 256, 3).astype('float32')
y = np.array([[1], [0]])

# Train a step
model.fit(x, y)

# Predict
print("Predictions:", model.predict(x))
