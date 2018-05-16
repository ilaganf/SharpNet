'''
dataset.py

Defines TensorFlow operations for loading,
processing, and batching the data

Possibly add later: data augmentation
'''

import tensorflow as tf


def input_op(filenames, params, is_training):
    '''
    Creates the dataset object to be run and used for training/evaluating

    Args:
        filenames: list of filenames (including path) of input images
        params: Config object that contains model hyperparameters
        is_training: boolean of whether or not we're training
    '''
    dataset = tf.data.Dataset.from_tensor_slices({'low-res': tf.constant(filenames),
                                                  'high-res': tf.constant(filenames)})
    
    # Load all the images from the filenames,
    # Properly re-size them all, then create low-res
    # versions of all the input images to use as training input
    dataset = dataset.map(parse_image_fn).map(low_res_fn)
    if is_training:
        # Shuffle all the input and repeat for unlimited epochs
        dataset = dataset.shuffle(len(filenames)).repeat()
    dataset = dataset.batch(params.batch_size).prefetch(1)

    iterator = dataset.make_initializable_iterator()
    images = iterator.get_next()
    iterator_init = iterator.initializer

    return images, iterator_init
    # return dataset.make_one_shot_iterator().get_next()


def parse_image_fn(fname1, fname2):
    '''
    Loads images from filenames
    '''
    img1 = _parse_helper(fname1)
    img2 = _parse_helper(fname2)
    return img1, img2


def _parse_helper(fname):
    image_string = tf.read_file(fname)
    decoded = tf.image.decode_image(image_string, channels=3)
    image = tf.image.convert_image_dtype(decoded, tf.float32)
    resized = tf.image.resize_images(image, [288,288])
    return resized


def low_res_fn(image, label):
    '''
    Applies gaussian blur, downscales image, then
    interpolates image back to original size
    '''
    blurred = image # TODO
    downscaled = tf.image.resize_images(blurred, [144, 144])
    upscaled = tf.image.resize_images(downscaled, [288, 288], method=ResizeMethod.BICUBIC)
    return upscaled, label
