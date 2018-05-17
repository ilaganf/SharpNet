'''
main.py

Run this file to load, train, 
and do whatever, coordinated with command line arguments

do_training: Coordinates loading in the data from ./data/, initializing
a model with the desired hyperparameters,
training the model, then writing the results and 
model checkpoints to ./experiments/
'''

import os

# Package imports go here
import tensorflow as tf

# Project imports go here
from architecture.model import Model

import architecture
import architecture.config as config
from architecture.run_training import train
from architecture.input import input_op
from architecture.model import EnhanceNet


# Credit to 224N course staff for CLI code
tf.app.flags.DEFINE_string("mode", "train", "Available modes: train / eval / predict")
tf.app.flags.DEFINE_string("experiment_name", "", "Unique name for your experiment. This will create a directory by this name in the experiments/ directory, which will hold all data related to this experiment")
tf.app.flags.DEFINE_integer("num_epochs", 0, "Number of epochs to train.")
tf.app.flags.DEFINE_string("ckpt_load_dir", "", "For eval and predict modes, which directory to load the checkpoint from.")

FLAGS = tf.app.flags.FLAGS


def do_prediction():
    pass


def do_evaluation():
    pass


def do_training():
    # Make experiments reproducible
    tf.set_random_seed(12345)

    param = config.Config("baseline")

    train_files = [os.path.join(config.TRAIN_DIR, f) for f in os.listdir(config.TRAIN_DIR)
                   if f.endswith('.jpg') or f.endswith('.png')]
    dev_files = [os.path.join(config.DEV_DIR, f) for f in os.listdir(config.DEV_DIR)
                   if f.endswith('.jpg') or f.endswith('.png')]
    param.train_size = len(train_files)
    param.dev_size = len(dev_files)

    train_data, train_initializer = input_op(train_files, param, is_training=True)
    dev_data, dev_initializer = input_op(train_files, param, is_training=False)

    train_model = EnhanceNet(train_data, train_initializer, param, is_training=True)
    dev_model = EnhanceNet(dev_data, dev_initializer, param, is_training=False)

    train(train_model, dev_model, param)


def main():
    


if __name__ == '__main__':
    main()