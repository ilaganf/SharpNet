#!/usr/bin/env python

import os

import tensorflow as tf
import scipy
import numpy as np
import architecture
import architecture.config as config
from architecture.input import input_op
from architecture.model import EnhanceNet

def evaluate(param, test_files):
    test_data, test_initializer = input_op(test_files, param, is_training=False)
    evaluate_model = EnhanceNet(test_data, test_initializer, param, is_training=False)
    evaluate_model.build_model()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, param.checkpoints)
        sess.run(evaluate_model.iter_init_op)
        ground_truth, input, pred = sess.run([evaluate_model.labels, evaluate_model.inputs
                                              , evaluate_model.prediction_op])       
    for i in range(ground_truth.shape[0]): 
        scipy.misc.imsave("evaluate/" + str(i) + "input.jpg", np.squeeze(input[i,:,:,:]))
        scipy.misc.imsave("evaluate/" + str(i) + "pred.jpg", np.squeeze(pred[i,:,:,:]))
        scipy.misc.imsave("evaluate/" + str(i) + "truth.jpg", np.squeeze(ground_truth[i,:,:,:]))
def main():
    tf.set_random_seed(12345)
    param = config.Config("baseline")
    
    test_files = [os.path.join(config.TEST_DIR, f) for f in os.listdir(config.TEST_DIR)
                  if f.endswith('.jpg') or f.endswith('.png')]
    
    evaluate(param, test_files)

if __name__=="__main__":
    main()
