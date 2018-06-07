'''
VAKNet_deep.py
'''

import tensorflow as tf

from architecture.VAKNet import VAKNet
from architecture.inception_resnet_v2 import ENDPOINTS


class VAKNetDeep(VAKNet):

    def add_prediction_op(self):
        with tf.variable_scope('model_prediction', reuse=tf.AUTO_REUSE):
            z = tf.layers.conv2d(self.input_data['low-res'], 64, 7, 1, padding='SAME',
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 name='conv_input')
            for i in range(10):
                block_name = "res_block%d"%i
                z = resid_block(block_name, z)
            out = tf.layers.conv2d(z, 3, 5, 1, padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                   activation=None, name='reconstruction_output')
        return out


    def get_ops(self):
        pred = self.pred + self.input_data['low-res']
        with tf.variable_scope("metrics", reuse=tf.AUTO_REUSE):
            labels = self.input_data['high-res']
            # SSIM
            ssim_op = tf.reduce_mean(tf.image.ssim(pred, 
                                               labels, max_val=1.0))
            # PSNR
            _labels = tf.reshape(labels, (self.config.batch_size, self.config.input_size[0], self.config.input_size[1], 3))
            _pred = tf.reshape(pred, (self.config.batch_size, self.config.input_size[0], self.config.input_size[1], 3))
            psnr_op = tf.reduce_mean(tf.image.psnr(_pred, _labels, max_val=1.0, name='psnr_op'))
    
            # MSE
            mse_op = tf.reduce_mean(tf.losses.mean_squared_error(pred, labels))

        if self.grad_norm is not None:
            return [ssim_op, psnr_op, mse_op, self.grad_norm]
        return [ssim_op, psnr_op, mse_op]


    def add_loss_op(self, pred):
        with tf.variable_scope('loss', reuse=tf.AUTO_REUSE):
            pred_img = pred + self.input_data['low-res']

            _, pred_end_points = self._resnet_activation(pred_img)
            _, target_points = self._resnet_activation(self.input_data['high-res'])

            # L2 loss for residual
            loss = tf.losses.mean_squared_error(
                labels=(self.input_data['low-res'] - self.input_data['high-res']),
                predictions=pred, reduction=tf.losses.Reduction.MEAN)

            # Sum up differences in representation over all activations
            for endpoint in ENDPOINTS:
                loss += tf.losses.mean_squared_error(
                          labels=target_points[endpoint],
                          predictions=pred_end_points[endpoint])
        return loss


def resid_block(blockname, z):
    with tf.variable_scope(blockname):
        with tf.variable_scope("Branch_0"):
            root = tf.layers.conv2d(z, filters=32, kernel_size=5, strides=1, padding="SAME",
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    name='conv2d_root')
        with tf.variable_scope("Branch_1"):
            branch1_0 = tf.layers.conv2d(z, 32, 1, 1, 'SAME', kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='conv2d_a')
            branch1_1 = tf.layers.conv2d(branch1_0, 32, 5, 1, 'SAME', kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='conv2d_b')
        with tf.variable_scope("Branch_2"):
            branch2_0 = tf.layers.conv2d(z, 32, 1, 1, 'SAME', kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='conv2d_a')
            branch2_1 = tf.layers.conv2d(branch2_0, 48, 5, 1, 'SAME', kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='conv2d_b')
            branch2_2 = tf.layers.conv2d(branch2_1, 64, 5, 1, 'SAME', kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='conv2d_c')
        combined = tf.concat(axis=3, values=[root, branch1_1, branch2_2])
        out = tf.layers.conv2d(combined, 64, 1, 1, 'SAME', kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               name='conv2d_combine')
        return out

