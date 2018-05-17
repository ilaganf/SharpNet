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
import architecture
import architecture.config as config
from architecture.run_training import train
from architecture.input import input_op
from architecture.model import EnhanceNet


# Credit to 224N course staff for CLI code
MAIN_DIR = os.path.relpath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # relative path of the main directory
EXPERIMENTS_DIR = os.path.join(MAIN_DIR, "experiments") # relative path of experiments dir

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


def main(unused_argv):
    # Print an error message if you've entered flags incorrectly
    if len(unused_argv) != 1:
        raise Exception("There is a problem with the entered flags: %s" % unused_argv)

    # Define train_dir
    if not FLAGS.experiment_name and FLAGS.mode != "eval" and FLAGS.mode != "predict":
        raise Exception("You need to specify --experiment_name")
    FLAGS.train_dir = FLAGS.train_dir or os.path.join(EXPERIMENTS_DIR, FLAGS.experiment_name)

    # Initialize bestmodel directory
    bestmodel_dir = os.path.join(FLAGS.train_dir, "best_checkpoint")

    # Different behavior based on mode
    if FLAGS.mode == 'train':
        do_training()
    elif FLAGS.mode == 'eval':
        pass
    elif FLAGS.mode == 'predict':
        pass
    else:
        raise Exception("Unexpected value of FLAGS.mode: %s" % FLAGS.mode)

if __name__ == '__main__':
    tf.app.run()