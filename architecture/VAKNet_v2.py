'''
VAKNet_v2.py
'''

import tensorflow as tf

from architecture.VAKNet import VAKNet

class VAKNetV2(VAKNet):
    def add_prediction_op(self):
        f1, f2, f3, f4, f5, f6, f7 = [7, 7, 7, 7, 7, 1, 5]
        n1, n2, n3, n4, n5, n6 = [64, 32, 16, 32, 64, 32]

        num_channels = 3
        with tf.variable_scope('model_prediction', reuse=tf.AUTO_REUSE):
            layer1_out = tf.layers.conv2d(inputs=self.input_data['low-res'], filters=n1, 
                                          kernel_size=f1, strides=1, padding='SAME', 
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          name='conv1')
            layer2_out = tf.layers.conv2d(inputs=layer1_out, filters=n2, 
                                          kernel_size=f2, strides=2, padding='SAME', 
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(), 
                                          name='conv2')
            layer3_out = tf.layers.conv2d(inputs=layer2_out, filters=n3,
                                          kernel_size=f3, strides=2, padding='SAME',
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          name='conv3')
            layer4_out = tf.layers.conv2d_transpose(inputs=layer3_out, filters=n4, kernel_size=f4,
                                                    strides=1, padding='SAME',
                                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                    name='transpose_conv4')
            layer5_out = tf.layers.conv2d_transpose(inputs=layer4_out, filters=n5, kernel_size=f5,
                                                    strides=1, padding='SAME',
                                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                    name='transpose_conv5')
            layer6_out = tf.layers.conv2d(inputs=layer5_out, filters=n6, kernel_size=f6,
                                          strides=1, padding='SAME',
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          name='conv6')
            reconstructed = tf.layers.conv2d(inputs=layer6_out, filters=num_channels, 
                                             kernel_size=f7, strides=1, padding='SAME', 
                                             kernel_initializer=tf.contrib.layers.xavier_initializer(), 
                                             activation=None, name='output')
        return reconstructed