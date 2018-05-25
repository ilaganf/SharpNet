
import os

import tensorflow as tf
import scipy
import numpy as np
import config
from input import input_op
from model import EnhanceNet

# Dataset to debug on
debug_dataset = config.TEST_DIR
model_name = "baseline"

def visual_debug(param, test_files):
    '''
    Provides a visual demonstrating the
    model's performance on a particular dataset.
    In particular, given a list of image names,
    saves the ground truth, distored, and predicted
    image inside the debug folder
    '''
    test_data, test_initializer = input_op(test_files, param, is_training=False)
    evaluate_model = EnhanceNet(test_data, test_initializer, param, is_training=False)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, param.checkpoints)
        sess.run(evaluate_model.iter_init_op)
        truth, distorted, pred = sess.run([evaluate_model.labels, 
                                           evaluate_model.inputs,
                                           evaluate_model.prediction_op])       
    for i in range(param.batch_size): 
        scipy.misc.imsave("./debug/" + str(i) + "distorted.jpg", np.squeeze(distorted[i,:,:,:]))
        scipy.misc.imsave("./debug/" + str(i) + "pred.jpg", np.squeeze(pred[i,:,:,:]))
        scipy.misc.imsave("./debug/" + str(i) + "truth.jpg", np.squeeze(truth[i,:,:,:]))


def main():
    tf.set_random_seed(12345)
    param = config.Config(model_name)
    test_files = [os.path.join(debug_dataset, f) for f in os.listdir(debug_dataset)
                  if f.endswith('.jpg') or f.endswith('.png')]
    visual_debug(param, test_files)

if __name__=="__main__":
    main()
