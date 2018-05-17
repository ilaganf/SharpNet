'''
evaluate.py

Creates a tensorflow session and runs the model
through the test set.
'''
import os

import tensorflow as tf
import scipy
import numpy as np
from input import input_op


def evaluate(model, params):
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(model.iter_init_op)
        saver.restore(sess, params.checkpoints)

        num_runs = (params.train_size + params.batch_size - 1) // params.batch_size
        total_loss = 0.0
        total_ssim = 0.0
        total_psnr = 0.0
        for i in range(num_runs)
            mse, ssim, psnr, pred = sess.run([model.loss_op, model.ssim_op, 
                                              model.psnr_op, model.prediction_op])
            total_loss += mse
            total_ssim += ssim
            total_psnr += psnr

        return total_loss/num_runs, total_ssim/num_runs, total_psnr/num_runs


def main():
    tf.set_random_seed(12345)
    param = config.Config("baseline")

    test_files = [os.path.join(config.TEST_DIR, f) for f in os.listdir(config.TEST_DIR)
                  if f.endswith('.jpg') or f.endswith('.png')]

    loss, ssim, psnr = evaluate(param, test_files)


if __name__=="__main__":
    main()
