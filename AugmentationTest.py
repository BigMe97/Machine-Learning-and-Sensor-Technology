import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pickle

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
ImageTest = x_train[0:5]
yTest = y_train[0:5]
# Display dataset size
print(f"Training samples: {ImageTest.shape[0]}")
print(f"Test samples: {ImageTest.shape[0]}")

# Visualize sample images from the dataset
plt.figure(figsize=(10, 4))
for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.imshow(ImageTest[i], cmap='gray')
    plt.title(f"Label: {yTest[i]}")
    plt.axis('off')
plt.show(block=False)


print('Create augmented data')
# Create an ImageDataGenerator with translation options
x_augmented = []
for i in range(len(ImageTest)):
    shiftedImage = np.roll(ImageTest[i], 3)
    x_augmented.append(shiftedImage)

x_augmented = np.array(x_augmented)

x_train_combined = np.concatenate((ImageTest, x_augmented), axis=0)
y_train_combined = np.append(yTest, y_test)

print(f"Training samples: {x_train_combined.shape[0]}")

plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train_combined[i], cmap='gray')
    plt.title(f"Label: {y_train_combined[i]}")
    plt.axis('off')
plt.show()
