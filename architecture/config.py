'''
config.py

File for defining object that is responsible
for holding hyperparameters for the model
'''
import os
import json

#################
# Data location #
#################

TRAIN_DIR = './data/train2017/'
DEV_DIR = './data/val2017/'
# TRAIN_DIR = './data/train_debug/'
# DEV_DIR = './data/val_debug'
TEST_DIR = './data/test2017/'
PRED_DIR_IN = './prediction_inputs/'
PRED_DIR_OUT = './prediction_outputs/'

######################
# Training constants #
######################
INPUT_SIZE = (299, 299)

######################
# Blurring constants #
######################
downscale_size = (144,144) #
kernel_length  = 3
gaussian_sig   = 1

HOME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(HOME, 'experiments')

class Config():

    def __init__(self, is_new, path, **kwargs):
        if is_new:
            # Naming and filesystem parameters
            self.experiment_name = kwargs.get('experiment_name', 'default')
            self.basepath = path
            self.checkpoint_dir = self.basepath+'/weights/'
            self.checkpoints = os.path.join(self.checkpoint_dir, 'weights.ckpt')
            if not os.path.exists(self.checkpoint_dir):
                os.mkdir(self.checkpoint_dir)
            self.tensorboard_dir = os.path.join(self.basepath, 'tensorboard/')
            if not os.path.exists(self.tensorboard_dir):
                os.mkdir(self.tensorboard_dir)
            self.save_every = kwargs.get('save_every', 1)

            # Learning-related hyperparameters
            self.learning_rate = kwargs.get('learning_rate', 0.01)
            self.num_epochs = kwargs.get('num_epochs', 5)
            self.batch_size = kwargs.get('batch_size', 16)
            self.max_grad_norm = kwargs.get('max_grad_norm', 0)
            self.grad_norm = None

            # Input parameters
            self.shuffle_buffer_size = kwargs.get('shuffle_buffer_size', 10000)
            self.input_size = kwargs.get('input_size', INPUT_SIZE)

            self.write_params(self.basepath+'/params.json')
        else:
            fname = os.path.join(path, 'params.json')
            self.load_params(fname)


    def load_params(self, filename):
        with open(filename) as f:
            data = json.load(f)
            for key in data.keys():
                self.__dict__[key] = data[key]


    def write_params(self, filename):
        # Create dict...
        dict_to_write = {}
        for key in self.__dict__.keys():
            dict_to_write[key] = self.__dict__[key]

        with open(filename, 'w') as f:
            json.dump(dict_to_write, f)
