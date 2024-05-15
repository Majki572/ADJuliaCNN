import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import psutil
import time

tf.config.set_visible_devices([], 'GPU')

def print_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"Memory Usage: {mem_info.rss / 1024**2:.2f} MiB")

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255


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

class MemoryAndTimeCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()
    
    def on_epoch_end(self, epoch, logs=None):
        elapsed_time = time.time() - self.start_time
        print(f"\nEpoch {epoch + 1}")
        print(f"Time: {elapsed_time:.2f} seconds")
        print_memory_usage()
        train_loss, train_accuracy = logs.get('loss'), logs.get('accuracy')
        val_loss, val_accuracy = logs.get('val_loss'), logs.get('val_accuracy')
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Test Loss: {val_loss:.4f}, Test Accuracy: {val_accuracy:.4f}")
        print()

model.fit(x_train, y_train, epochs=3, batch_size=100, validation_data=(x_test, y_test), callbacks=[MemoryAndTimeCallback()])


# Results:
# Epoch 1/3
# 591/600 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8169 - loss: 0.6366    
# Epoch 1
# Time: 2.08 seconds
# Memory Usage: 754.15 MiB
# Train Loss: 0.2954, Train Accuracy: 0.9146
# Test Loss: 0.1005, Test Accuracy: 0.9691

# 600/600 ━━━━━━━━━━━━━━━━━━━━ 2s 2ms/step - accuracy: 0.8185 - loss: 0.6309 - val_accuracy: 0.9691 - val_loss: 0.1005
# Epoch 2/3
# 589/600 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9676 - loss: 0.1095 
# Epoch 2
# Time: 1.39 seconds
# Memory Usage: 756.66 MiB
# Train Loss: 0.0986, Train Accuracy: 0.9701
# Test Loss: 0.0855, Test Accuracy: 0.9722

# 600/600 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.9677 - loss: 0.1093 - val_accuracy: 0.9722 - val_loss: 0.0855
# Epoch 3/3
# 598/600 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9749 - loss: 0.0789 
# Epoch 3
# Time: 1.35 seconds
# Memory Usage: 756.78 MiB
# Train Loss: 0.0745, Train Accuracy: 0.9761
# Test Loss: 0.0686, Test Accuracy: 0.9772

# 600/600 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.9749 - loss: 0.0789 - val_accuracy: 0.9772 - val_loss: 0.0686