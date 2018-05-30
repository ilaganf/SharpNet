'''
VakNet.py
'''
import os

import tensorflow as tf

from inception_resnet_v2 import inception_resnet_v2_base, inception_resnet_v2_arg_scope
import config


class VAKNet(EnhanceNet):

    def add_prediction_op(self):
        with tf.variable_scope('model_prediction', reuse=tf.AUTO_REUSE):
            f1, f2, f3 = [9, 1, 5] # Size of filter kernels
            n1, n2 = [64, 32] #Number of filters in layer
            num_channels = 3
            
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
            _, end_points = self._resnet_activation(reconstructed)
            final_out = end_points['Mixed_6a']
            self.loss_activation = final_out
            return reconstructed

    
    def add_loss_op(self, pred):
        '''
        Adds the loss function to the graph
        '''
        with tf.variable_scope('model_loss', reuse=tf.AUTO_REUSE):
            loss = tf.losses.mean_squared_error(
                       labels=self.input_data['high-res'],
                       predictions=pred,
                       reduction=tf.losses.Reduction.MEAN)

            _, target_points = self._resnet_activation(self.input_data['high-res'])
            target = target_points['Mixed_6a']
            loss += tf.losses.mean_squared_error(
                        labels=target, predictions=self.loss_activation,
                        reduction=tf.losses.Reduction.MEAN)
        return loss

    
    def add_training_op(self, loss):
        optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
        self.global_step = tf.train.get_or_create_global_step()
        return optimizer.minimize(loss, global_step=self.global_step)


    def _resnet_activation(self, reconstructed):
        assert config.INPUT_SIZE == (299, 299), \
            "If using Inception Resnet, need 299x299 input images"

        with slim.arg_scope(inception_resnet_v2_arg_scope()):
             logits, end_points = inception_resnet_v2(reconstructed, is_training=False)
        return logits, end_points


    def fit(self, train_data, val_data):
        '''
        Runs training/validation loop

        @train_data and @val_data are pandas dataframes
        '''
        train_iter_init, val_iter_init, self.input_data = self._prepare_train_val(train_data, val_data)
        self.build()
        saver = tf.train.Saver(max_to_keep=1)
        resnet_saver = tf.train.Saver()
        train_len = len(train_data)
        val_len = len(val_data)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            train_writer = tf.summary.FileWriter(self.config.tensorboard_dir + self.config.experiment_name + '/train/')
            val_writer = tf.summary.FileWriter(self.config.tensorboard_dir + self.config.experiment_name + '/val/')

            # Restore weights for inception_resnet v2
            assert os.path.exists('./inception_resnet_v2_2016_08_30.ckpt'), \
                    "You need to make sure the inception_resnet weights are in the same directory as this model"
            resnet_saver.restore(sess, './inception_resnet_v2_2016_08_30.ckpt')

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
                        print("New best MSE! Saving model in {}".format(self.config.checkpoints))
                    saver.save(sess, self.config.checkpoints)
                if self.verbose: print()

        