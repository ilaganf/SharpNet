'''
VakNet.py
'''
import os

from tensorflow.python import pywrap_tensorflow
import tensorflow as tf

from architecture.inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
from architecture.config import INPUT_SIZE
from architecture.core.Model import Model


class VAKNet(Model):
    '''
    Vaknet, which adds the feature loss from inception_resnet v2
    '''
    def add_prediction_op(self):
        f1, f2, f3 = [9, 1, 5] # Size of filter kernels
        n1, n2 = [64, 32] #Number of filters in layer
        num_channels = 3
        with tf.variable_scope('model_prediction', reuse=tf.AUTO_REUSE):
            layer1_out = tf.layers.conv2d(inputs=self.input_data['low-res'], filters=n1, 
                                          kernel_size=f1, strides=1, padding='SAME', 
                                          kernel_initializer=tf.random_normal_initializer(0, 0.001),
                                          bias_initializer=tf.constant_initializer(0),
                                          name='conv1')
            layer2_out = tf.layers.conv2d(inputs=layer1_out, filters=n2, 
                                          kernel_size=f2, strides=1, padding='SAME', 
                                          kernel_initializer=tf.random_normal_initializer(0, 0.001), 
                                          bias_initializer=tf.constant_initializer(0),
                                          name='conv2')
            reconstructed = tf.layers.conv2d(inputs=layer2_out, filters=num_channels, 
                                             kernel_size=f3, strides=1, padding='SAME', 
                                             kernel_initializer=tf.random_normal_initializer(0, 0.001), 
                                             bias_initializer=tf.constant_initializer(0), 
                                             activation=None, name='output')
        return reconstructed

    
    def add_loss_op(self, pred):
        '''
        Adds the loss function to the graph
        '''
        with tf.variable_scope('loss', reuse=tf.AUTO_REUSE):
            _, end_points = self._resnet_activation(pred)
            pred_activation = end_points['Mixed_6a']
            
            loss = tf.losses.mean_squared_error(
                labels=self.input_data['high-res'],
                predictions=pred,
                reduction=tf.losses.Reduction.MEAN)
            
            _, target_points = self._resnet_activation(self.input_data['high-res'])
            target = target_points['Mixed_6a']
            loss += tf.losses.mean_squared_error(
                        labels=target, predictions=pred_activation,
                        reduction=tf.losses.Reduction.MEAN)
        return loss

    
    def add_training_op(self, loss):
        with tf.variable_scope("training", reuse=tf.AUTO_REUSE):
            optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
            self.global_step = tf.train.get_or_create_global_step()
            grads_and_vars = optimizer.compute_gradients(loss)
            trainable_grads_and_vars = [g_v for g_v in grads_and_vars \
                                        if "Inception" not in g_v[1].name]
            train_grads = [g[0] for g in trainable_grads_and_vars]
            train_vars = [v[1] for v in trainable_grads_and_vars]
            if self.config.max_grad_norm != 0:
                train_grads, _ = tf.clip_by_global_norm(train_grads, self.config.max_grad_norm)
            self.grad_norm = tf.global_norm(train_grads)
            train_op = optimizer.apply_gradients(zip(train_grads, train_vars), global_step=self.global_step)

            assert self.grad_norm is not None, "Grad norms were set incorrectly"

        return train_op


    def _resnet_activation(self, reconstructed):
        assert INPUT_SIZE == (299, 299), \
            "If using Inception Resnet, need 299x299 input images"

        with tf.contrib.slim.arg_scope(inception_resnet_v2_arg_scope()):
             logits, end_points = inception_resnet_v2(reconstructed, is_training=False)
        return logits, end_points


    def fit(self, train_data, val_data, load=False):
        '''
        Runs training/validation loop

        @train_data and @val_data are pandas dataframes
        '''
        train_iter_init, val_iter_init, self.input_data = self._prepare_train_val(train_data, val_data)
        self.build()
        saver = tf.train.Saver(max_to_keep=1)
        train_len = len(train_data)
        val_len = len(val_data)
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # Restore weights for inception_resnet v2
            assert os.path.exists('architecture/inception_resnet_v2_2016_08_30.ckpt'), \
                "You need to make sure the inception_resnet weights are in the same directory as this model"
            optimistic_restore(sess, 'architecture/inception_resnet_v2_2016_08_30.ckpt', graph=sess.graph)
            
            train_writer = tf.summary.FileWriter(self.config.tensorboard_dir + '/train/', graph=sess.graph)
            val_writer = tf.summary.FileWriter(self.config.tensorboard_dir + '/val/', graph=sess.graph)
            if load:
                print("Restoring weights from {}".format(self.config.checkpoint_dir))
                saver.restore(sess, self.config.checkpoints)

            training_handle = sess.run(train_iter_init.string_handle())
            val_handle = sess.run(val_iter_init.string_handle())
            
            best_loss = float("inf")
            for epoch in range(self.config.num_epochs):
                if self.verbose:
                    print("Epoch {}\n".format(epoch+1))
                sess.run(train_iter_init.initializer)        
                sess.run(val_iter_init.initializer)
                train_metrics = self.run_epoch(sess, training_handle, train_writer, "train", train_len)
                val_metrics = self.run_epoch(sess, val_handle, val_writer, "val", val_len)
                train_metrics = [("Train_" + x[0], x[1]) for x in train_metrics]
                val_metrics = [("Val_" + x[0], x[1]) for x in val_metrics]
                
                val_loss = val_metrics[0][1]
                if val_loss < best_loss:
                    best_loss = val_loss
                    if self.verbose:
                        print("New best MSE! Saving model in {}".format(self.config.checkpoint_dir))
                    saver.save(sess, self.config.checkpoints)
                if self.verbose: print()

        
def inception_arg_scope(weight_decay=0.00004,
                        use_batch_norm=True,
                        batch_norm_decay=0.9997,
                        batch_norm_epsilon=0.001):
  """Defines the default arg scope for inception models.
  Args:
    weight_decay: The weight decay to use for regularizing the model.
    use_batch_norm: "If `True`, batch_norm is applied after each convolution.
    batch_norm_decay: Decay for batch norm moving average.
    batch_norm_epsilon: Small float added to variance to avoid dividing by zero
      in batch norm.
  Returns:
    An `arg_scope` to use for the inception models.
  """
  batch_norm_params = {
      # Decay for the moving averages.
      'decay': batch_norm_decay,
      # epsilon to prevent 0s in variance.
      'epsilon': batch_norm_epsilon,
      # collection containing update_ops.
      'updates_collections': tf.GraphKeys.UPDATE_OPS,
  }
  if use_batch_norm:
    normalizer_fn = tf.contrib.slim.batch_norm
    normalizer_params = batch_norm_params
  else:
    normalizer_fn = None
    normalizer_params = {}
  # Set weight_decay for weights in Conv and FC layers.
  with tf.contrib.slim.arg_scope([tf.contrib.slim.conv2d, tf.contrib.slim.fully_connected],
                      weights_regularizer=tf.contrib.slim.l2_regularizer(weight_decay)):
    with tf.contrib.slim.arg_scope(
        [tf.contrib.slim.conv2d],
        weights_initializer=tf.contrib.slim.variance_scaling_initializer(),
        activation_fn=tf.nn.relu,
        normalizer_fn=normalizer_fn,
        normalizer_params=normalizer_params) as sc:
      return sc


def optimistic_restore(session, save_file, graph=tf.get_default_graph()):
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
            if var.name.split(':')[0] in saved_shapes])    
    restore_vars = []    
    for var_name, saved_var_name in var_names:            
        curr_var = graph.get_tensor_by_name(var_name)
        var_shape = curr_var.get_shape().as_list()
        if var_shape == saved_shapes[saved_var_name]:
            restore_vars.append(curr_var)
    opt_saver = tf.train.Saver(restore_vars)
    opt_saver.restore(session, save_file)
