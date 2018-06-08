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
import numpy as np
import scipy

# Project imports go here
import architecture
import architecture.config as config
#from architecture.run_training import train
#from architecture.evaluate import evaluate
#from architecture.input import input_op
from architecture.EnhanceNet import EnhanceNet
from architecture.VAKNet import VAKNet
from architecture.VAKNet_v2 import VAKNetV2, VAKNetV2L1, VAKNetV2Resid, VAKNetV2Features
from architecture.VAKNetDeep import VAKNetDeep

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


def do_prediction(params):
    pred_files = [os.path.join(config.PRED_DIR_IN, f) for f in os.listdir(config.PRED_DIR_IN)
                  if f.endswith('.jpg')]

    # Depending on the model, might need to add extra code to add residuals back to the low_res input
    #test_model = EnhanceNet(params)
    #test_model = VAKNet(params)
    #test_model = VAKNetV2(params)
    #test_model = VAKNetV2Resid(params)
    test_model = VAKNetDeep(params)
    
    outputs = test_model.predict(pred_files)
    input_imgs = []
    for file in pred_files:
        input_imgs.append(scipy.ndimage.imread(file))
    input_imgs = np.array(input_imgs)
    #outputs = input_imgs - 255*outputs
    outputs *= 255
    
    input = tf.placeholder(tf.uint8)
    encode_op = tf.image.encode_jpeg(input, quality=100)

    with tf.Session() as sess:
        for i, img in enumerate(outputs):
            encoded = sess.run(encode_op, {input: img})
            with open(config.PRED_DIR_OUT + str(i) + '.jpeg', 'wb') as f:
                f.write(encoded)


def do_evaluation(params):
    test_files = [os.path.join(config.TEST_DIR, f) for f in os.listdir(config.TEST_DIR)
                  if f.endswith('.jpg')]#[:20000]
    #test_model = VAKNetV2(params)
    #test_model = EnhanceNet(params)
    #test_model = VAKNet(params)
    #test_model = VAKNetV2Resid(params)
    test_model = VAKNetDeep(params)
    
    mse, ssim, psnr = test_model.evaluate(test_files) # TODO: return some sample images
    mse_mean = np.mean(mse)
    mse_std = np.std(mse)
    ssim_mean = np.mean(ssim)
    ssim_std = np.std(ssim)
    psnr_mean = np.mean(psnr)
    psnr_std = np.std(psnr)
    print("MSE Mean:", mse_mean)
    print("MSE Std:", mse_std)
    print("SSIM Mean:", ssim_mean)
    print("SSIM Std:", ssim_std)
    print("PSNR Mean:", psnr_mean)
    print("PSNR Std:", psnr_std)
    
    
def do_training(params, load_weights=False):
    # Make experiments reproducible
    tf.set_random_seed(12345)

    train_files = [os.path.join(config.TRAIN_DIR, f) for f in os.listdir(config.TRAIN_DIR)
                   if f.endswith('.jpg')][:60000]
    dev_files = [os.path.join(config.DEV_DIR, f) for f in os.listdir(config.DEV_DIR)
                   if f.endswith('.jpg')]#[:5000]
    # model = EnhanceNet(params)
    # model = VAKNet(params)
    # model = VAKNetV2(params)
    # model = VAKNetV2L1(params)
    # model = VAKNetV2Resid(params)
    # model = VAKNetV2Features(params)
    model = VAKNetDeep(params)
    model.fit(train_files, dev_files, load_weights)


def main(unused_argv):

    # Print an error message if you've entered flags incorrectly
    if len(unused_argv) != 1:
        raise Exception("There is a problem with the entered flags: %s" % unused_argv)

    if not FLAGS.experiment_name:
            raise Exception("You need to specify --experiment_name")

    # Define train_dir
    train_dir = os.path.join(EXPERIMENTS_DIR, FLAGS.experiment_name)
    
    if FLAGS.mode == 'train':
        if os.path.exists(os.path.join(train_dir, 'weights/')):
            print("Mode is 'train', but weights for experiment '{}' already exist.".format(FLAGS.experiment_name))
            instruction = input("Do you want to proceed and overwrite? (y/n): ")
            while instruction.strip().lower() not in ['y', 'n']:
                instruction = input("Please enter y or n: ")
            if instruction.strip().lower() == 'n':
                print("Abort mission")
                return
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
        do_training(params, FLAGS.warm_start)
    elif FLAGS.mode == 'eval':
         print("Mode is 'eval': ignoring all other flags besides --experiment_name")
         print("Loading parameters and weights from {}".format(train_dir))
         params = config.Config(is_new=False, path=train_dir)
         do_evaluation(params)
    elif FLAGS.mode == 'predict':
        print("Mode is 'predict': ignoring all other flags besides --experiment_name")
        print("Loading parameters and weights from {}".format(train_dir))
        params = config.Config(is_new=False, path=train_dir)
        do_prediction(params)
    else:
        raise Exception("Unexpected value of FLAGS.mode: %s" % FLAGS.mode)


if __name__ == '__main__':
    tf.app.run()
