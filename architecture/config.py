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

######
# Training constants
#

######################
# Blurring constants #
######################
downscale_size = (144,144)
kernel_length  = 3
gaussian_sig   = 1

class Config():

    def __init__(self, exp_name, **kwargs):
        self.experiment_name = exp_name
        self.basepath = './experiments/'+exp_name
        self.checkpoints = self.basepath+'/weights/'
        if not os.path.exists(self.basepath):
            os.mkdir(self.basepath)
            os.mkdir(self.checkpoints)

        self.save_every = 1
        self.learning_rate = 0.01
        self.num_epochs = 5
        self.batch_size = 16
        # temp - until we build file architecture, need these params
        for key, val in kwargs.items():
            self.__dict__[key] = val
        self.write_params(self.basepath+'/params.json')
        self.train_size = None
        self.dev_size = None


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
