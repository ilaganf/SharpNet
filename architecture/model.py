'''
model.py

Defines the model to train and make predictions

Will implement as a class that contains each of the
necessary operations
'''
import tensorflow as tf
import numpy as np
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
        print(self.labels)
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
            
            layer1_out = tf.layers.conv2d(inputs=self.inputs, filters=n1, 
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
        labels = tf.reshape(self.labels, (16, 288, 288, 3))
        pred = tf.reshape(self.prediction_op, (16, 288, 288, 3))
        return tf.image.psnr(pred, labels, max_val=1.0, name='psnr_op')

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
        self.variable_summaries()

    def variable_summaries(self):
        #tf.summary.scalar("Mean Squared Error", self.mse_op)
        
        with tf.variable_scope("metrics"):
            #tf.summary.scalar("Peak Signal-to-Noise Ratio", self.psnr_op)
            #tf.summary.scalar("Structural Similarity", self.ssim_op)
            tf.summary.scalar("Loss", self.loss_op)
            weights = [var for var in tf.trainable_variables() if 'model_prediction' in str(var)]
            l2 = np.sum([tf.nn.l2_loss(var) for var in weights])
            tf.summary.scalar("L2 norm", l2)
            self.merged = tf.summary.merge_all()
