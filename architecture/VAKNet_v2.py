'''
VAKNet_v2.py
'''

import tensorflow as tf

from architecture.VAKNet import VAKNet
from architecture.inception_resnet_v2 import ENDPOINTS

class VAKNetV2(VAKNet):
    '''
    Vaknet v2, which has a deeper downsampling-upsampling reconstruction
    network 
    '''
    def add_prediction_op(self):
        f1, f2, f3, f4, f5, f6, f7 = [7, 7, 7, 7, 7, 1, 2]
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
                                                    strides=2, padding='SAME',
                                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                    name='transpose_conv4')
            layer5_out = tf.layers.conv2d_transpose(inputs=layer4_out, filters=n5, kernel_size=f5,
                                                    strides=2, padding='SAME',
                                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                    name='transpose_conv5')
            layer6_out = tf.layers.conv2d(inputs=layer5_out, filters=n6, kernel_size=f6,
                                          strides=1, padding='SAME',
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          name='conv6')
            reconstructed = tf.layers.conv2d(inputs=layer6_out, filters=num_channels, 
                                             kernel_size=f7, strides=1, padding='VALID', 
                                             kernel_initializer=tf.contrib.layers.xavier_initializer(), 
                                             activation=None, name='output')
        return reconstructed


class VAKNetV2L1(VAKNetV2):
    '''
    Replaces the per-pixel L2 with per-pixel L1 loss
    '''
    def add_loss_op(self, pred):
        '''
        Adds the loss function to the graph
        '''
        with tf.variable_scope('loss', reuse=tf.AUTO_REUSE):
            _, end_points = self._resnet_activation(pred)
            pred_activation = end_points['Mixed_6a']
            
            loss = tf.losses.absolute_difference(
                labels=self.input_data['high-res'],
                predictions=pred,
                reduction=tf.losses.Reduction.MEAN)
            
            _, target_points = self._resnet_activation(self.input_data['high-res'])
            target = target_points['Mixed_6a']
            loss += tf.losses.mean_squared_error(
                        labels=target, predictions=pred_activation,
                        reduction=tf.losses.Reduction.MEAN)
        return loss


class VAKNetV2Resid(VAKNetV2):
    '''
    Learns the resdiduals instead of the full image
    '''
    def add_loss_op(self, pred):
        with tf.variable_scope('loss', reuse=tf.AUTO_REUSE):
            pred_img = pred + self.input_data['low-res']

            _, pred_end_points = self._resnet_activation(pred_img)
            _, target_points = self._resnet_activation(self.input_data['high-res'])

            # L1 loss for residual
            loss = tf.losses.absolute_difference(
                labels=(self.input_data['high-res'] - self.input_data['low-res']),
                predictions=pred, reduction=tf.losses.Reduction.MEAN)

            # Sum up differences in representation over all activations
            for endpoint in ENDPOINTS:
                loss += tf.losses.mean_squared_error(
                          labels=target_points[endpoint],
                          predictions=pred_end_points[endpoint])
        return loss


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


class VAKNetV2Features(VAKNetV2):
    '''
    Learns the residuals but only uses the summed feature vectors as
    the loss function
    '''
    def add_loss_op(self, pred):
        with tf.variable_scope('loss', reuse=tf.AUTO_REUSE):
            pred_img = pred + self.input_data['low-res']

            _, pred_end_points = self._resnet_activation(pred_img)
            _, target_points = self._resnet_activation(self.input_data['high-res'])

            # Sum up differences in representation over all activations
            loss = tf.constant(0.0)
            for endpoint in ENDPOINTS:
                loss += tf.losses.mean_squared_error(
                          labels=target_points[endpoint],
                          predictions=pred_end_points[endpoint])
        return loss


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

