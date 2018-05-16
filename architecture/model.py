'''
model.py

Defines the model to train and make predictions

Will implement as a class that contains each of the
necessary operations
'''
import tensorflow as tf

import architecture.config

class EnhanceNet():
    '''
    Object that defines all the necessary operations
    so they can be readily accessed for training/testing
    '''

    def __init__(self, inputs, inputs_initializer, config, is_training=True):

        # Iterator over dataset object
        self.inputs = inputs['low-res']
        self.labels = inputs['high-res']
        self.iter_init_op = inputs_initializer
        
        # Config object that holds hyperparameters
        self.config = config

        # Internal objects
        self.prediction_op = None
        self.loss_op = None
        self.optimizer_op = None
        self.mse_op = None
        self.is_training = is_training
        
    def add_prediction_op(self):
        with tf.variable_scope('model_prediction', reuse=(not self.is_training)):
            f1, f2, f3 = [9, 1, 5] # Size of filter kernels
            n1, n2 = [64, 32] #Number of filters in layer
            num_channels = 3

            layer1_out = tf.layers.conv2d(inputs=self.inputs, num_outputs=n1, 
                                                  kernel_size=f1, 
                                                  stride=1, padding='SAME', 
                                                  weights_initializer=tf.random_normal_initializer(0, 0.001),
                                                  biases_initializer=tf.constant_initializer(0))
            layer2_out = tf.contrib.layers.conv2d(inputs=layer1_out, num_outputs=n2, 
                                                  kernel_size=[f1, f1, n1], 
                                                  stride=1, padding='SAME', 
                                                  weights_initializer=tf.random_normal_initializer(0, 0.001), 
                                                  biases_initializer=tf.constant_initializer(0))
            final_out = tf.contrib.layers.conv2d(inputs=layer2_out, num_outputs=num_channels, 
                                                 kernel_size=[f1, f1, n2], 
                                                 stride=1, padding='SAME', 
                                                 weights_initializer=tf.random_normal_initializer(0, 0.001), 
                                                 biases_initializer=tf.constant_initializer(0), 
                                                 activation_fn=None)
            return final_out

    def add_loss_op(self):
        # self.labels has y vals
        with tf.variable_scope('model_loss', reuse=not self.is_training):
            loss = tf.losses.mean_squared_error(
                labels=self.labels,
                predictions=self.prediction_op,
                reduction=tf.losses.Reduction.MEAN
            )
            return loss

    def add_psnr_op(self):
        return tf.image.psnr(self.prediction_op, self.labels, max_val=1.0, name='psnr_op')

    def add_ssim_op(self):
        return tf.image.ssim(self.prediction_op, self.labels, max_val=1.0)

    def add_mse_op(self):
        mse, _ = tf.metrics.mean_squared_error(self.labels, self.prediction_op, 
                                               name='mse_metric')
        return mse

    def add_training_op(self):
        optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
        global_step = tf.train.get_or_create_global_step()
        return optimizer.minimize(self.loss_op, global_step=global_step)

    def build_model(self):
        self.prediction_op = self.add_prediction_op()
        self.psnr_op = self.add_psnr_op()
        self.ssim_op = self.add_ssim_op()
        self.mse_op = self.add_mse_op()
        self.loss_op = self.add_loss_op()
        if self.is_training:
            self.train_op = self.add_training_op()
        
