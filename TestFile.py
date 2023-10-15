import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import tensorflow as tf
from tensorflow import keras
import pickle

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# Display dataset size
print(f"Training samples: {x_train.shape[0]}")
print(f"Test samples: {x_test.shape[0]}")

print('Creating augmented data...')
# Create a translated dataset
x_augmented = []
for i in range(len(x_train)):
    shiftedImage = np.roll(x_train[i], 3)
    x_augmented.append(shiftedImage)
    if i % 1000 == 0:
        print('.', end='', flush=True)

x_augmented = np.array(x_augmented)

x_train_combined = np.concatenate((x_train, x_augmented), axis=0)
y_train_combined = np.append(y_train, y_train)

x_train_combined = x_train_combined / 255.0

print('.')
print('Training model...')

# Create and train a Decision Tree classifier
model = DecisionTreeClassifier(random_state=42)
model.fit(x_train_combined.reshape(x_train_combined.shape[0], -1), y_train_combined)

# Make predictions on the test data
print('Testing model...')
y_pred = model.predict(x_test.reshape(x_test.shape[0], -1))

# Evaluate the model
test_accuracy = (y_pred == y_test).mean()

print(f'Test accuracy: {test_accuracy}')

# Save the model
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print('Finished')

plt.plot(y_pred - y_test)
plt.title('Predictions difference')
plt.grid()
plt.yticks(range(-9, 10))
plt.show()


