import os
import numpy as np
from tensorflow.python import pywrap_tensorflow
import tensorflow as tf
from tensorflow.python.tools import inspect_checkpoint as chkp
from inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope

def _resnet_activation(reconstructed):

    with tf.contrib.slim.arg_scope(inception_resnet_v2_arg_scope()):
        logits, end_points = inception_resnet_v2(reconstructed, is_training=False)
    return logits, end_points


if __name__=="__main__":
    
    x = tf.constant(np.zeros((16,299,299,3)), tf.float32)
    hello = _resnet_activation(x)
    resnet_saver = tf.train.Saver()
    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())
        # Restore weights for inception_resnet v2
        assert os.path.exists('architecture/inception_resnet_v2_2016_08_30.ckpt'), \
            "You need to make sure the inception_resnet weights are in the same directory as this model"
        #checkpoint_path = os.path.join(model_dir, "model.ckpt")
        #reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
        #var_to_shape_map = reader.get_variable_to_shape_map()
        #for key in var_to_shape_map:
        
        resnet_saver.restore(sess, 'architecture/inception_resnet_v2_2016_08_30.ckpt')
