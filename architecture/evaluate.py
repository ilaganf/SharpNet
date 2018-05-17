'''
evaluate.py

Creates a tensorflow session and runs the model
through the test set.
'''
import os

import tensorflow as tf
import scipy
import numpy as np
import architecture.config as config
from architecture.input import input_op
from architecture.model import EnhanceNet

def evaluate(param, test_files, evaluate_model):
    test_data, test_initializer = input_op(test_files, param, is_training=False)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(evaluate_model.iter_init_op)
        saver.restore(sess, param.checkpoints)

        num_runs = params.test_size / params.eval_size
        total_loss = 0.0
        total_ssim = 0.0
        total_psnr = 0.0
        for i in range(num_runs)
            ground_truth, input, pred = sess.run([evaluate_model.labels, evaluate_model.inputs, evaluate_model.prediction_op])
            total_loss += evaluate_model.add_loss_op()
            total_ssim += evaluate_model.add_ssim_op()
            total_psnr += evaluate_model.add_psnr_op()
        return total_loss/num_runs, total_ssim/num_runs, total_psnr/num_runs


def main():
    tf.set_random_seed(12345)
    param = config.Config("baseline")

    test_files = [os.path.join(config.TEST_DIR, f) for f in os.listdir(config.TEST_DIR)
                  if f.endswith('.jpg') or f.endswith('.png')]

    loss, ssim, psnr = evaluate(param, test_files)

if __name__=="__main__":
    main()
