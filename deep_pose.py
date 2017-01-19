from __future__ import print_function

import tensorflow as tf
import numpy as np
import util
import lsp_dataset

# Parameters
learning_rate = 0.0005
training_iters = 400000
batch_size = 2
display_step = 10

# Network Parameters
n_input = 220  # data input (img shape: 28*28)
n_depth = 3  # total classes (0-9 digits)
n_classes = 28  # total classes (0-9 digits)
dropout = 0.75  # Dropout, probability to keep units

with tf.name_scope('input'):
    # tf Graph input
    x = tf.placeholder(tf.float32, [None, n_input, n_input, n_depth])
with tf.name_scope('output'):
    y = tf.placeholder(tf.float32, [None, n_classes])

pred = util.inference(x)

def get_loss(pred, y):
    label_validity = tf.cast(tf.sign(y, name='label_validity'), tf.float32)
    labels_float = tf.cast(y, tf.float32)
    minus_op = tf.sub(pred, labels_float, name='Diff_Op')
    abs_op = tf.abs(minus_op, name='Abs_Op')
    loss_values = tf.mul(label_validity, abs_op, name='lossValues')
    loss_mean = tf.reduce_mean(loss_values, name='MeanPixelError')
    tf.add_to_collection('losses', loss_mean)
    return tf.add_n(tf.get_collection('losses'), 'total_loss'), loss_mean

cost, loss_mean = get_loss(pred, y)

tf.summary.scalar('cost', cost)

optimizer = util.train(cost)

merged = tf.summary.merge_all()

# Initializing the variables
init = tf.initialize_all_variables()

train_x, train_y, test_x, test_y = lsp_dataset.get_dataset()

# Launch the graph
with tf.Session() as sess:
    train_writer = tf.train.SummaryWriter('train_logs/deep_pose',
                                          sess.graph)
    test_writer = tf.train.SummaryWriter('test_logs/deep_pose',
                                         sess.graph)
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:

        train_idx = np.random.randint(len(train_x), size=batch_size)
        batch_x = train_x[train_idx]
        batch_y = train_y[train_idx]

        op, summary = sess.run(
            [optimizer, merged], feed_dict={x: batch_x, y: batch_y})
        train_writer.add_summary(summary, step)

        if step % display_step == 0:
            # Calculate batch loss and accuracy
            summary, loss, ret_loss_mean, ret_pred = sess.run([merged, cost, loss_mean, pred], feed_dict={x: test_x,
                                                                                          y: test_y})
            test_writer.add_summary(summary, step)
            print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Loss Mean= " + \
                  "{:.5f}".format(ret_loss_mean))
            # print(test_y - ret_pred)
            # print(ret_pred)
            # print(test_y)
            # print(np.add(test_y, np.negative(ret_pred)))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 256 mnist test images
    print("Testing Loss Mean:", \
          sess.run(loss_mean, feed_dict={x: test_x,
                                         y: test_y}))
