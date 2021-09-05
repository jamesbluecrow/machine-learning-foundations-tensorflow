# https://colab.research.google.com/github/lmoroney/mlday-tokyo/blob/master/Lab4-Using-Convolutions.ipynb

# 1. Add imports
import tensorflow as tf

print(tf.__version__)

# 2. Prepare & Standard data
mnist = tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

training_images = training_images.reshape(60000, 28, 28, 1)
training_images = training_images / 255

test_images = test_images.reshape(10000, 28, 28, 1)
test_images = test_images / 255

# 3. Prepare model & add neurons
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=[28, 28, 1]),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])


# 4. Prepare callback to finish training early
class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (logs.get("accuracy") >= 0.97):
            print("Finish training early when accuracy => 97%")
            self.model.stop_training = True


# 5. Train neural network
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(training_images, training_labels, epochs=10, callbacks=[CustomCallback()])

# 6. Test
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(test_accuracy)
