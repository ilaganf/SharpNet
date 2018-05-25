
import tensorflow as tf
import scipy.stats as st
import numpy as np
import architecture.core.utils as utils

class Model():
    def __init__(self, config, verbose=True):
        self.config = config
        self.verbose = verbose
    
    def get_ops(self):
        pred = self.pred
        labels = self.input_data['high-res']
        # SSIM
        ssim_op = tf.reduce_mean(tf.image.ssim(pred, 
                                               labels, max_val=1.0))
        # PSNR
        _labels = tf.reshape(labels, (self.config.batch_size, 288, 288, 3))
        _pred = tf.reshape(pred, (self.config.batch_size, 288, 288, 3))
        psnr_op = tf.reduce_mean(tf.image.psnr(_pred, _labels, max_val=1.0, name='psnr_op'))
    
        # MSE
        mse_op = tf.reduce_mean(tf.losses.mean_squared_error(pred, labels))
        return [ssim_op, psnr_op, mse_op]

    
    def build(self):
        '''
        Adds important graph functions as well as the global 
        step which is important for logging training progress
        '''
        self.global_step = tf.Variable(0, trainable=False)
        self.is_training = tf.placeholder(tf.bool)
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)    
        self.ops = self.get_ops()
   
        
    def input_op(self, data):
        '''
        Creates the dataset object to be run and used for training/evaluating
        
        Args:
        filenames: list of filenames (including path) of input images
        params: Config object that contains model hyperparameters
        is_training: boolean of wheter 
        '''
        dataset = tf.data.Dataset.from_tensor_slices({'low-res': tf.constant(data),
                                                  'high-res': tf.constant(data)})
        
        # Load all the images from the filenames,
        # Properly re-size them all, then create low-res
        # versions of all the input images to use as training input
        dataset = dataset.shuffle(len(data)).repeat()

        dataset = dataset.map(lambda x: utils.parse_image_fn(x))
        dataset = dataset.shuffle(self.config.num_shuffle_buffer).repeat().prefetch(1)
        dataset = dataset.batch(self.config.batch_size)
        return dataset
        
    
    def add_prediction_op(self):
        '''
        Adds the workhouse prediction structure to the graph.
        You need to set self.input_data to point to the correct
        input data before calling this function.
        '''
        raise NotImplementedError("Do not substantiate a base Model class")

    
    def add_loss_op(self, pred):
        '''
        Adds the loss function to the graph
        '''
        raise NotImplementedError("Do not substantiate a base Model class")

    
    def add_train_op(self, loss):
        '''
        Adds the training operations (optimizer and gradient
        clipping operations for instance) to the graph
        '''
        raise NotImplementedError("Do not substantiate a base Model class")

      
    def fit(self, train_data, val_data):
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
            train_writer = tf.summary.FileWriter(self.config.tensorboard_dir + self.config.experiment_name + '/train/')
            val_writer = tf.summary.FileWriter(self.config.tensorboard_dir + self.config.experiment_name + '/val/')

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
                        print("New best MSE! Saving model in ./results/model.weights/weights")
                    saver.save(sess, "./results/trained_variables.ckpt")
                if self.verbose: print()

                
    def predict(self, data):
        '''
        @data is a pandas dataframe
        '''
        saver = tf.train.Saver(max_to_keep = 1)
        input = self.input_op(data)
        iterator = input.make_initializable_iterator()
        self.input_data = iterator.get_next()
        with tf.Session() as sess:
            saver.restore(sess, './results/trained_variables.ckpt')
            sess.run(tf.global_variables_initializer())
            test_handle = sess.run(iterator.string_handle())
            sess.run(iterator.initializer)
            preds = []
            while True:
                try:
                    _pred = sess.run(self.pred, feed_dict={self.handle: test_handle,
                                                           self.is_training: False})
                    preds.append(_pred)
                except tf.errors.OutOfRangeError:
                    break
        return np.concatenate(preds)

    
    def _prepare_train_val(self, train_data, dev_data):
        train_input, val_input = self.input_op(train_data), self.input_op(dev_data)
        train_iter_init = train_input.make_initializable_iterator()
        val_iter_init = val_input.make_initializable_iterator()
        self.handle = tf.placeholder(tf.string, shape=())
        iterator = tf.data.Iterator.from_string_handle(self.handle,
                                                       train_input.output_types,
                                                       train_input.output_shapes)
        return train_iter_init, val_iter_init, iterator.get_next()

    
    def run_epoch(self, sess, handle, writer, epoch_type, data_len):
        metrics = []
        ops = [self.loss, self.pred, self.input_data, self.global_step] + self.get_ops()
        if epoch_type == "train": ops += [self.train_op] 
        bar = tf.keras.utils.Progbar(target=data_len)
        while True:
            try:
                output = sess.run(ops, feed_dict={self.handle: handle, self.is_training: epoch_type == "train"})
                metrics.append(output[0:len(ops) - (epoch_type=="train") - 1])
            except tf.errors.OutOfRangeError:
                break
            metric = (("Loss", output[0]), 
                                 ("Squared Error", output[6]), 
                                 ("PSNR", output[4]), 
                                 ("SSIM", output[5]))
            bar.add(self.config.batch_size, values=metric)
        summary, important_metrics = self.make_summary(metrics, epoch_type)
        writer.add_summary(summary, global_step)
        return important_metrics

    
    def make_summary(self, metrics, epoch_type):
        loss = np.mean([x[0] for x in metrics])
        pred = np.concatenate([x[1] for x in metrics])
        labels = np.concatenate([x[2]['high-res'] for x in metrics])
        psnr = np.concatenate([x[4] for x in metrics])
        ssim = np.concatenate([x[5] for x in metrics])
        mse = np.concatenate([x[6] for x in metrics])
        summary = tf.Summary(value=[tf.Summary.Value(tag='Loss', simple_value=loss),
                                    tf.Summary.Value(tag="Squared Error", simple_value=mse),
                                    tf.Summary.Value(tag='Peak Signal-to-Noise Ratio', simple_value=psnr),
                                    tf.Summary.Value(tag='Structural Similarity', simple_value=ssim)]) 
        important_metrics = (("Loss", loss), ("Squared Error", mse), ("PSNR", psnr), ("SSIM", ssim))
        return summary, important_metrics
   
    
        
    def weight_l2(self):
        weights = [var for var in tf.trainable_variables() if 'weights' in str(var)]
        l2 = np.sum([tf.nn.l2_loss(var) for var in weights])
        return l2
