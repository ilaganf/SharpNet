'''
model.py

Defines the model to train and make predictions

Will implement as a class that contains each of the
necessary operations
'''
import tensorflow as tf

import config

class EnhanceNet():
    '''
    Object that defines all the necessary operations
    so they can be readily accessed for training/testing
    '''
    def add_prediction_op(self):
        with tf.variable.scope('model_prediction'):
            f1, f2, f3 = [9, 1, 5] # Size of filter kernels
            n1, n2 = [64, 32] #Number of filters in layer
            num_channels = 3

            layer1_out = tf.contrib.layers.conv2d(inputs=self.input, num_outputs=n1, kernel_size = [f1, f1, num_channels], stride=1, padding='SAME', weights_initializer = tf.random_normal_initializer(0, 0.001), biases_initializer=tf.constant_initializer(0))
            layer2_out = tf.contrib.layers.conv2d(inputs=layer1_out, num_outputs=n2, kernel_size = [f1, f1, n1], stride=1, padding='SAME', weights_initializer = tf.random_normal_initializer(0, 0.001), biases_initializer=tf.constant_initializer(0))
            final_out = tf.contrib.layers.conv2d(inputs=layer2_out, num_outputs=num_channels, kernel_size = [f1, f1, n2], stride=1, padding='SAME', weights_initializer = tf.random_normal_initializer(0, 0.001), biases_initializer=tf.constant_initializer(0), activation_fn=None)
            return final_out

    def add_loss_op(self, prediction_in):
        # self.labels has y vals
        with tf.variable.scope('model_loss'):
            loss = tf.losses.mean_squared_error(
                labels=self.labels,
                predictions=prediction_in,
                reduction=tf.losses.Reduction.MEAN
            )
            return loss

    def __init__(self):
        pass
