import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import psutil
import time

def print_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"Memory Usage: {mem_info.rss / 1024**2:.2f} MiB")

physical_devices = tf.config.experimental.list_physical_devices()
print("Physical devices:", physical_devices)

visible_devices = tf.config.experimental.list_logical_devices()
print("Visible devices:", visible_devices)

tf.debugging.set_log_device_placement(True)

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
        print(f"Epoch {epoch + 1}")
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
# 595/600 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7803 - loss: 0.7537Epoch 1
# Time: 2.25 seconds
# Memory Usage: 753.74 MiB
# Train Loss: 0.3451, Train Accuracy: 0.9001
# Test Loss: 0.1059, Test Accuracy: 0.9667

# 600/600 ━━━━━━━━━━━━━━━━━━━━ 2s 3ms/step - accuracy: 0.7815 - loss: 0.7497 - val_accuracy: 0.9667 - val_loss: 0.1059
# Epoch 2/3
# 581/600 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9668 - loss: 0.1080Epoch 2
# Time: 1.40 seconds
# Memory Usage: 758.04 MiB
# Train Loss: 0.0979, Train Accuracy: 0.9699
# Test Loss: 0.0783, Test Accuracy: 0.9756

# 600/600 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.9669 - loss: 0.1077 - val_accuracy: 0.9756 - val_loss: 0.0783
# Epoch 3/3
# 599/600 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9754 - loss: 0.0793Epoch 3
# Time: 1.37 seconds
# Memory Usage: 758.24 MiB
# Train Loss: 0.0723, Train Accuracy: 0.9775
# Test Loss: 0.0543, Test Accuracy: 0.9819

# 600/600 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.9754 - loss: 0.0793 - val_accuracy: 0.9819 - val_loss: 0.0543