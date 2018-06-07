'''
Utility functions for use in the model.
For the most part provides functions for
reading in images, distorting them and
saving distorted images
'''

import tensorflow as tf
import numpy as np
import scipy.stats as st
import architecture
import scipy
import architecture.config as config


def read_image(fname):
    '''
    Reads image into a tensor
    fname: filepath to image
    '''
    image_string = tf.read_file(fname)
    decoded = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(decoded, tf.float32)
    resized = tf.image.resize_images(image, config.INPUT_SIZE)
    return resized


def distort_image(image):
    '''
    Applies gaussian blur, downscales image, then
    interpolates image back to original size
    '''

    # Creates the gaussian kernel credit to antonilo in the github
    # project TensBlur for this code snippet
    def gauss_kernel(kernlen=config.kernel_length, 
                     nsig=config.gaussian_sig, channels=3):
        '''
        Creates the gaussian kernel for convolution over the 
        '''
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
    blurred = tf.nn.depthwise_conv2d(image, filter, strides=[1,1,1,1], padding="SAME", name="gaussian_blur")
    downscaled = tf.image.resize_images(blurred, config.downscale_size, method=tf.image.ResizeMethod.BICUBIC)
    upscaled = tf.image.resize_images(downscaled, config.INPUT_SIZE, method=tf.image.ResizeMethod.BICUBIC)
    upscaled = tf.squeeze(upscaled, axis=0)
    return upscaled


def save_distorted_image(fname_in, fname_out):
    '''
    Given an input image saves a distorted version
    of the image according to our blurring method.
    Serves as a method to ensure that our distortion
    method works.
    '''
    image = scipy.ndimage.imread(fname_in)
    image_placeholder = tf.placeholder(dtype=tf.float32, shape=image.shape)
    blur_op = tf.squeeze(distort_image(image_placeholder))
    with tf.Session() as sess:
        blurred_image = sess.run(blur_op, feed_dict={image_placeholder: image})
    scipy.misc.imsave(fname_out, blurred_image)
     
    
def parse_image_fn(fnames):
    '''
    Helper function for input operations.
    Loads images from filenames
    '''
    n_low_res, n_high_res = fnames['low-res'], fnames['high-res']
    low_res = read_image(n_low_res)
    low_res = distort_image(low_res)
    high_res = read_image(n_high_res)
    return {"low-res":low_res, "high-res":high_res}

if __name__=="__main__":
    save_distorted_image("./0_low_res.jpg", "./ok.jpg")
