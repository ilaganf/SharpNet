
import os

import tensorflow as tf
import scipy
import numpy as np
import architecture.config as config
from architecture.VAKNet_v2 import VAKNetV2 as Model

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

    model = Model(param)
    input_data = model.input_op(test_files, epoch_type="val")
    iterator = input_data.make_one_shot_iterator()
    model.input_data = iterator.get_next()
    model.build()
    saver = tf.train.Saver(max_to_keep = 1)
    with tf.Session() as sess:
        saver.restore(sess, model.config.checkpoints)
        input_images = sess.run([model.input_data])

    low_res = input_images[0]['low-res']
    high_res = input_images[0]['high-res']
        
    for i in range(param.batch_size): 
        scipy.misc.imsave("./debug/low_res/" + str(i) + "_low_res.jpg", np.squeeze(low_res[i,:,:,:]))
        scipy.misc.imsave("./debug/high_res/" + str(i) + "_high_res.jpg", np.squeeze(high_res[i,:,:,:]))
        print(np.sum(low_res[i,:,:,:] - high_res[i,:,:,:]))


def main():
    tf.set_random_seed(12345)
    param = config.Config(is_new=False, path="./experiments/vaknet_v2")
    test_files = [os.path.join(debug_dataset, f) for f in os.listdir(debug_dataset)
                  if f.endswith('.jpg') or f.endswith('.png')]
    visual_debug(param, test_files)

if __name__=="__main__":
    main()
