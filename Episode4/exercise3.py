# https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Exercises/Exercise%203%20-%20Convolutions/Exercise%203%20-%20Question.ipynb
# In the videos you looked at how you would improve Fashion MNIST using Convolutions.
# For your exercise see if you can improve MNIST to 99.8% accuracy or more using only a single convolutional layer
# and a single MaxPooling 2D. You should stop training once the accuracy goes above this amount.
# It should happen in less than 20 epochs, so it's ok to hard code the number of epochs for training,
# but your training must end once it hits the above metric. If it doesn't, then you'll need to redesign your layers.
#
# When 99.8% accuracy has been hit, you should print out the string "Reached 99.8% accuracy so cancelling training!"

import tensorflow as tf


mnist = tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()