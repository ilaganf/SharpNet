'''
model.py

Defines the model to train and make predictions

Will implement as a class that contains each of the
necessary operations
'''
import tensorflow as tf
import numpy as np
from architecture.core.Model import Model

class EnhanceNet(Model): 
    def add_prediction_op(self):
        with tf.variable_scope('model_prediction', reuse=tf.AUTO_REUSE):
            f1, f2, f3 = [9, 1, 5] # Size of filter kernels
            n1, n2 = [64, 32] #Number of filters in layer
            num_channels = 3
            
            layer1_out = tf.layers.conv2d(inputs=self.input_data['low-res'], filters=n1, 
                                          kernel_size=f1, strides=1, padding='SAME', 
                                          kernel_initializer=tf.random_normal_initializer(0, 0.001),
                                          bias_initializer=tf.constant_initializer(0),
                                          name='conv1')
            layer2_out = tf.layers.conv2d(inputs=layer1_out, filters=n2, 
                                          kernel_size=f2, strides=1, padding='SAME', 
                                          kernel_initializer=tf.random_normal_initializer(0, 0.001), 
                                          bias_initializer=tf.constant_initializer(0),
                                          name='conv2')
            final_out = tf.layers.conv2d(inputs=layer2_out, filters=num_channels, 
                                         kernel_size=f3, strides=1, padding='SAME', 
                                         kernel_initializer=tf.random_normal_initializer(0, 0.001), 
                                         bias_initializer=tf.constant_initializer(0), 
                                         activation=None, name='output')
            return final_out


    def add_loss_op(self, pred):
        # self.labels has y vals
        with tf.variable_scope('model_loss', reuse=tf.AUTO_REUSE):
            loss = tf.losses.mean_squared_error(
                labels=self.input_data['high-res'],
                predictions=pred,
                reduction=tf.losses.Reduction.MEAN
            )
            return loss


    def add_training_op(self, loss):
        optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
        self.global_step = tf.train.get_or_create_global_step()
        return optimizer.minimize(loss, global_step=self.global_step)


class VAKNet(EnhanceNet):

    def foo():
        # The full bells and whistles model
        pass
