'''
run_training.py

Creates a tensorflow session and actually
runs the training loop
'''
import tensorflow as tf

def train(train_model, dev_model):
    saver = tf.train.Saver(max_to_keep=1)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # TODO: Tensorboard integration
        for epoch in range(train_model.config.num_epochs):
            num_steps = (train_size + params.batch_size - 1) // params.batch_size
