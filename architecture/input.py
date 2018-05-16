'''
dataset.py

Defines TensorFlow operations for loading,
processing, and batching the data

Possibly add later: data augmentation
'''

import tensorflow as tf
from PIL import Image
import numpy as np
import scipy.stats as st
import scipy
import architecture.config as config

test_image = "../data/test2017/000000000001.jpg"

def input_op(filenames, params, is_training):
    '''
    Creates the dataset object to be run and used for training/evaluating

    Args:
        filenames: list of filenames (including path) of input images
        params: Config object that contains model hyperparameters
        is_training: boolean of wheter 
    '''
    dataset = tf.data.Dataset.from_tensor_slices({'low-res': tf.constant(filenames),
                                                  'high-res': tf.constant(filenames)})
    
    # Load all the images from the filenames,
    # Properly re-size them all, then create low-res
    # versions of all the input images to use as training input
    print(dataset)
    if is_training:
        # Shuffle all the input and repeat for unlimited epochs
        dataset = dataset.shuffle(len(filenames)).repeat()

    dataset = dataset.map(lambda x: parse_image_fn(x))
    dataset = dataset.batch(params.batch_size).prefetch(1)

    iterator = dataset.make_initializable_iterator()
    images = iterator.get_next()
    print(images)
    iterator_init = iterator.initializer

    return images, iterator_init


def parse_image_fn(fnames):
    '''
    Loads images from filenames
    '''
    n_low_res, n_high_res = fnames['low-res'], fnames['high-res']
    low_res = _parse_helper(n_low_res)
    low_res = low_res_fn(low_res)
    high_res = _parse_helper(n_high_res)
    return {"low-res":low_res, "high-res":high_res}


def _parse_helper(fname):
    image_string = tf.read_file(fname)
    decoded = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(decoded, tf.float32)
    resized = tf.image.resize_images(image, [288,288])
    return resized


def low_res_fn(image):
    '''
    Applies gaussian blur, downscales image, then
    interpolates image back to original size
    '''

    # Creating the gaussian kernel
    # credit to antonilo in the github
    # project TensBlur for this code snippet
    def gauss_kernel(kernlen=config.kernel_length, nsig=config.gaussian_sig, channels=3):
        interval = (2*nsig+1.)/(kernlen)
        x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
        kern1d = np.diff(st.norm.cdf(x))
        kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
        kernel = kernel_raw/kernel_raw.sum()
        out_filter = np.array(kernel, dtype = np.float32)
        out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
        out_filter = np.repeat(out_filter, channels, axis = 2)
        return out_filter
    
    filter = tf.constant(gauss_kernel())
    image = tf.expand_dims(image, axis=0)
    blurred = tf.nn.conv2d(image, filter, strides=[1,1,1,1], padding="SAME", name="gaussian_blur")
    downscaled = tf.image.resize_images(blurred, config.downscale_size, method=tf.image.ResizeMethod.BICUBIC)
    upscaled = tf.image.resize_images(downscaled, [288, 288], method=tf.image.ResizeMethod.BICUBIC)
    upscaled = tf.squeeze(upscaled)
    return upscaled

if __name__=="__main__":
    file = scipy.ndimage.imread(test_image)
    file = np.expand_dims(file, axis=0)
    image_placeholder = tf.placeholder(dtype=tf.float32, shape=((1, 480, 640, 3)))
    upscaled, label = low_res_fn(image_placeholder, None)
    with tf.Session() as sess:
        fd = {image_placeholder:file}
        output = sess.run(upscaled, feed_dict=fd)
    output = np.squeeze(output)
    print(output.shape)
    scipy.misc.imsave("my_img.jpg", output)
