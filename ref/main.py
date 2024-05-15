import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape data to fit the model (28, 28, 1) for grayscale images
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255

# One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = Sequential([
    Conv2D(6, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(16, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(84, activation='relu'),
    Dense(10, activation='linear')
])

model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))

for epoch in range(3):
    train_loss, train_accuracy = history.history['loss'][epoch], history.history['accuracy'][epoch]
    test_loss, test_accuracy = history.history['val_loss'][epoch], history.history['val_accuracy'][epoch]
    print(f"Epoch {epoch + 1}")
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    print()

# Results:
# Epoch 1/3
# 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 4s 2ms/step - accuracy: 0.8678 - loss: 0.4307 - val_accuracy: 0.9774 - val_loss: 0.0704
# Epoch 2/3
# 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 3s 1ms/step - accuracy: 0.9777 - loss: 0.0743 - val_accuracy: 0.9831 - val_loss: 0.0514
# Epoch 3/3
# 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 3s 1ms/step - accuracy: 0.9835 - loss: 0.0517 - val_accuracy: 0.9835 - val_loss: 0.0529
# Epoch 1
# Train Loss: 0.2001, Train Accuracy: 0.9396
# Test Loss: 0.0704, Test Accuracy: 0.9774

# Epoch 2
# Train Loss: 0.0696, Train Accuracy: 0.9791
# Test Loss: 0.0514, Test Accuracy: 0.9831

# Epoch 3
# Train Loss: 0.0507, Train Accuracy: 0.9840
# Test Loss: 0.0529, Test Accuracy: 0.9835