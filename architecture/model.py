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

    def __init__(self, inputs, config, is_training=True):

        # Iterator over dataset object
        self.inputs = inputs[0]
        self.labels = inputs[1]

        # Config object that holds hyperparameters
        self.config = config


    def add_training_op(self, loss):
        optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
        global_step = tf.train.get_or_create_global_step()
        return optimizer.minimize(loss, global_step=global_step, scope='model_prediction')
