'''
model.py

Defines the model (to be passed to tf.estimator.Estimator)
to train and make predictions

Will mostly be using tensorflow functional API since that's
what plays well with Estimator
'''
import tensorflow as tf

import config

def enhance_net(input):
    '''
    Returns function that defines the tensorflow model:
    model_fn(features, labels, mode, config)
    '''
    pass
