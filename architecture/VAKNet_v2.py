'''
VAKNet_v2.py
'''

import tensorflow as tf

from architecture.VAKNet import VAKNet

class VAKNetV2(VAKNet):
    def add_prediction_op(self):
        f1, f2, f3 = [9, 1, 5] # Size of filter kernels
        n1, n2 = [64, 32] #Number of filters in layer
        num_channels = 3
        with tf.variable_scope('model_prediction', reuse=tf.AUTO_REUSE):
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
            reconstructed = tf.layers.conv2d(inputs=layer2_out, filters=num_channels, 
                                             kernel_size=f3, strides=1, padding='SAME', 
                                             kernel_initializer=tf.random_normal_initializer(0, 0.001), 
                                             bias_initializer=tf.constant_initializer(0), 
                                             activation=None, name='output')
        return reconstructed