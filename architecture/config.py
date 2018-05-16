'''
config.py

File for defining object that is responsible
for holding hyperparameters for the model
'''
import os
import json

class Config():

    def __init__(self, exp_name):
        self.experiment_name = exp_name
        self.basepath = './experiments/'+exp_name
        self.checkpoints = self.basepath+'weights/'
        if not os.path.exists(self.basepath):
            os.mkdir(self.basepath)
            os.mkdir(self.checkpoints)
        self.write_params(self.basepath+'params.json')


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
