import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# Display dataset size
print(f"Training samples: {x_train.shape[0]}")
print(f"Test samples: {x_test.shape[0]}")

# Visualize sample images from the dataset
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i], cmap='gray')
    plt.title(f"Label: {y_train[i]}")
    plt.axis('off')
plt.show(block=False)

# Visualize the distribution of digit labels in the training set
plt.figure(figsize=(8, 5))
sns.countplot(x=y_test)
plt.title("Distribution of Digit Labels in Training Set")
plt.xlabel("Digit")
plt.ylabel("Count")
plt.show(block=False)


print('Creating augmented data...')
# Create an translated dataset
x_augmented = []
for i in range(len(x_train)):
    shiftedImage = np.roll(x_train[i], 3)
    x_augmented.append(shiftedImage)
    if i%1000==0:
        print('.', end='', flush=True)

x_augmented = np.array(x_augmented)

x_train_combined = np.concatenate((x_train, x_augmented), axis=0)
y_train_combined = np.append(y_train, y_train)


print('.')
print('Training model...')
# Create and train a model
model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
    ])
# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Train the model
model.fit(x_train_combined, y_train_combined, epochs=7)

# Make predictions on the test data
print('Testing model...')
y_pred = model.predict(x_test)

# Evaluate the model

test_accuracy = 0
y_predictions = np.zeros(x_test.shape[0])
for i in range(len(y_predictions)):
    y_predictions[i] = np.argmax(y_pred[i])
    if y_test[i] == y_predictions[i]:
        test_accuracy += 1

print(test_accuracy)
test_accuracy = test_accuracy/len(y_pred)

plt.figure(3)
plt.plot(y_predictions - y_test)
plt.title('Predictions difference')
plt.grid()
plt.ylabel('Digit')
plt.xlabel('Sample')
plt.yticks(range(-9, 10))

print(f'Test accuracy: {test_accuracy}')

# Save the model
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print('Finished')
plt.show()
