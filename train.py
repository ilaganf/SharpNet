'''
train.py

Coordinates loading in the data from ./data/, initializing
a model with the desired hyperparameters,
training the model, then writing the results and 
model checkpoints to ./experiments/
'''
import os

import tensorflow as tf

import architecture
import architecture.config as config
from architecture.run_training import train
from architecture.input import input_op
from architecture.model import EnhanceNet

def load_data():
    pass


def create_model():
    pass

def main():

    # Make experiments reproducible
    tf.set_random_seed(12345)

    param = config.Config("test")

    # logging
    # TODO

    train_files = [os.path.join(config.TRAIN_DIR, f) for f in os.listdir(config.TRAIN_DIR)
                   if f.endswith('.jpg') or f.endswith('.png')]
    dev_files = [os.path.join(config.DEV_DIR, f) for f in os.listdir(config.DEV_DIR)
                   if f.endswith('.jpg') or f.endswith('.png')]

    train_data, train_initializer = input_op(train_files, param, is_training=True)
    dev_data, dev_initializer = input_op(train_files, param, is_training=False)

    
    train_model = EnhanceNet(train_data, train_initializer, param, is_training=True)
    train_model.build_model()
    dev_model = EnhanceNet(dev_data, dev_initializer, param, is_training=False)
    dev_model.build_model()

    param.train_size = len(train_files)
    param.dev_size = len(dev_files)
    train(train_model, dev_model, param)


if __name__ == '__main__':
    main()
