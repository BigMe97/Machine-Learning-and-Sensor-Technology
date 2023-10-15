import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import pickle
np.set_printoptions(linewidth=600)
# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize the dataset
x_test = x_test / 255.0

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)


# Make predictions
# predictions = model.predict(x_test)

predictions = model.predict(x_test.reshape(len(x_test), -1))

y_predictions = np.zeros(x_test.shape[0])
print(len(y_predictions))
for i in range(len(y_predictions)):
    y_predictions[i] = np.argmax(predictions[i])
    # print(y_predictions[i])

print('Test data')
print(y_test)
print('Predictions')
print(y_predictions)

plt.plot(y_predictions)
plt.title('Predictions difference')
plt.grid()
plt.legend(['Error'])
plt.yticks(range(-9, 10))

print('Finished')
plt.show()

