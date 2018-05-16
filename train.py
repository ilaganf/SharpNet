'''
train.py

Coordinates loading in the data from ./data/, initializing
a model with the desired hyperparameters,
training the model, then writing the results and 
model checkpoints to ./experiments/
'''
import os

import tensorflow as tf

from architecture.config import Config
from architecture.input import input_op

TRAIN_DIR = './data/train/'
DEV_DIR = './data/dev/'


def load_data():
    pass


def create_model():
    pass


def train():
    pass


def main():

    # Make experiments reproducible
    tf.set_random_seed(12345)

    config = Config() # Load from some file eventually

    # logging
    # TODO

    train_files = [os.path.join(TRAIN_DIR, f) for f in os.listdir(TRAIN_DIR)
                   if f.endswith('.jpg') or f.endswith('.png')]
    dev_files = [os.path.join(TRAIN_DIR, f) for f in os.listdir(DEV_DIR)
                   if f.endswith('.jpg') or f.endswith('.png')]

    train_data, train_initializer = input_op(train_files, config, is_training=True)
    dev_data, dev_initializer = input_op(train_files, config, is_training=False)



if __name__ == '__main__':
    main()