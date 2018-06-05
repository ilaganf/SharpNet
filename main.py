'''
main.py

Run this file to load, train, 
and do whatever, coordinated with command line arguments

do_training: Coordinates loading in the data from ./data/, initializing
a model with the desired hyperparameters,
training the model, then writing the results and 
model checkpoints to ./experiments/
'''

# Standard Library imports go here
import os
import json
import shutil

# Package imports go here
import tensorflow as tf

# Project imports go here
import architecture
import architecture.config as config
#from architecture.run_training import train
#from architecture.evaluate import evaluate
#from architecture.input import input_op
from architecture.EnhanceNet import EnhanceNet
from architecture.VAKNet import VAKNet
from architecture.VAKNet_v2 import VAKNetV2, VAKNetV2L1, VAKNetV2Resid

# Credit to 224N course staff for CLI code
MAIN_DIR = os.path.relpath(os.path.dirname(os.path.abspath(__file__))) # relative path of the main directory
EXPERIMENTS_DIR = os.path.join(MAIN_DIR, "experiments") # relative path of experiments dir

tf.app.flags.DEFINE_string("mode", "train", "Available modes: train / eval / predict")
tf.app.flags.DEFINE_string("experiment_name", "", "Unique name for your experiment. This will create a directory by this name in the experiments/ directory, which will hold all data related to this experiment")
tf.app.flags.DEFINE_integer("num_epochs", 5, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("save_every", 1, "Number of epochs between saving")
tf.app.flags.DEFINE_float("learning_rate", .01, "Learning rate of the model")
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size to use")
tf.app.flags.DEFINE_integer("shuffle_buffer_size", 10000, "Size of shuffle buffer")
tf.app.flags.DEFINE_string("load_params", "", "Directory from which to load params, if they've already been made. If set, causes other parameters besides experiment name to be ignored.")
tf.app.flags.DEFINE_float("max_grad_norm", 10.0, "Max norm at which to clip gradients. 0 turns off gradient clipping")
tf.app.flags.DEFINE_bool("warm_start", False, "Whether or not to load existing weights. Only valid if load_params exists")

FLAGS = tf.app.flags.FLAGS


def do_prediction():
    pass


def do_evaluation(params):
    test_files = [os.path.join(config.TEST_DIR, f) for f in os.listdir(config.TEST_DIR)
                  if f.endswith('.jpg')]
    test_data, test_initializer = input_op(test_files, params, is_training=False)

    test_model = EnhanceNet(test_data, test_initializer, params, is_training=False)

    avg_loss, avg_ssim, avg_psnr = evaluate(test_model, params) # TODO: return some sample images


def do_training(params, load_weights=False):
    # Make experiments reproducible
    tf.set_random_seed(12345)

    train_files = [os.path.join(config.TRAIN_DIR, f) for f in os.listdir(config.TRAIN_DIR)
                   if f.endswith('.jpg')][:30000]
    dev_files = [os.path.join(config.DEV_DIR, f) for f in os.listdir(config.DEV_DIR)
                   if f.endswith('.jpg')][:3000]
    # model = EnhanceNet(params)
    # model = VAKNet(params)
    # model = VAKNetV2(params)
    # model = VAKNetV2L1(params)
    model = VAKNetV2Resid(params)
    model.fit(train_files, dev_files, load_weights)


def main(unused_argv):

    # Print an error message if you've entered flags incorrectly
    if len(unused_argv) != 1:
        raise Exception("There is a problem with the entered flags: %s" % unused_argv)

    # Define train_dir
    if not FLAGS.experiment_name and FLAGS.mode != "eval" and FLAGS.mode != "predict":
        raise Exception("You need to specify --experiment_name")
    train_dir = os.path.join(EXPERIMENTS_DIR, FLAGS.experiment_name)

    if FLAGS.load_params:
        params = config.Config(is_new=False, path=FLAGS.load_params)
        if os.path.exists(train_dir) and not FLAGS.warm_start:
            shutil.rmtree(params.tensorboard_dir)
            os.mkdir(os.path.join(params.basepath, 'tensorboard/'))
    else:
        # Load FLAGS into dict to save as a Config object later
        config_dict = {'experiment_name':FLAGS.experiment_name,
                       'save_every':FLAGS.save_every, 'num_epochs':FLAGS.num_epochs,
                       'learning_rate':FLAGS.learning_rate, 'batch_size':FLAGS.batch_size,
                       'shuffle_buffer_size':FLAGS.shuffle_buffer_size,
                       'max_grad_norm':FLAGS.max_grad_norm}
        if os.path.exists(train_dir):
            tens = os.path.join(train_dir, 'tensorboard')
            if os.path.exists(tens):
                shutil.rmtree(tens)
        else:
            os.mkdir(train_dir)
        params = config.Config(is_new=True, path=train_dir, **config_dict)

    # Different behavior based on mode
    if FLAGS.mode == 'train':
        do_training(params, FLAGS.warm_start)
    elif FLAGS.mode == 'eval':
        pass
    elif FLAGS.mode == 'predict':
        pass
    else:
        raise Exception("Unexpected value of FLAGS.mode: %s" % FLAGS.mode)


if __name__ == '__main__':
    tf.app.run()
