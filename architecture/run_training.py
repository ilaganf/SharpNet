'''
run_training.py

Creates a tensorflow session and actually
runs the training loop
'''
import tensorflow as tf


def train(train_model, dev_model, config):
    saver = tf.train.Saver(max_to_keep=1)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(train_model.iter_init_op)

        # TODO: Tensorboard integration

        best_dev = float('inf')
        for epoch in range(train_model.config.num_epochs):
            num_steps = (config.train_size + config.batch_size - 1) // config.batch_size

            for t in range(num_steps):
                prediction, loss = train_step(train_model, config, sess)
                if t % 10 == 0:
                    print("Iteration {} loss: {}".format(t, loss))
        saver.save(sess, config.checkpoints, global_step=epoch+1)

        if epoch % config.save_every == 0:
            num_steps = (config.eval_size + config.batch_size - 1) // config.batch_size

            for t in range(num_steps):




def train_step(model, config):
    global_step = tf.train.get_global_step()
    _, high_res, loss, global_step_val = sess.run([model.train_op, model.prediction_op,
                                                   model.loss_op, global_step])
    return high_res, loss