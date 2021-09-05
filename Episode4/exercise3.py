# https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Exercises/Exercise%203%20-%20Convolutions/Exercise%203%20-%20Question.ipynb
# In the videos you looked at how you would improve Fashion MNIST using Convolutions.
# For your exercise see if you can improve MNIST to 99.8% accuracy or more using only a single convolutional layer
# and a single MaxPooling 2D. You should stop training once the accuracy goes above this amount.
# It should happen in less than 20 epochs, so it's ok to hard code the number of epochs for training,
# but your training must end once it hits the above metric. If it doesn't, then you'll need to redesign your layers.
#
# When 99.8% accuracy has been hit, you should print out the string "Reached 99.8% accuracy so cancelling training!"

import tensorflow as tf

print(tf.__version__)

# 1. Load data
mnist = tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# 2. Reshape input data to include filter parameter / color depth & normalize data
training_images = training_images.reshape(60000, 28, 28, 1)
training_images = training_images / 255
test_images = test_images.reshape(10000, 28, 28, 1)
test_images = test_images / 255

# 4. Prepare model and add neurons
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=[28, 28, 1]),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])


# 5. Declare callback to stop training
class StopTrainingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get("accuracy") >= 0.998:
            print("Reached 99.8% accuracy so cancelling training!")
            self.model.stop_training = True


# 6. Train neural network
callbacks = [StopTrainingCallback()]
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(training_images, training_labels, epochs=19, callbacks=callbacks)

# 7. Test
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(test_accuracy)
