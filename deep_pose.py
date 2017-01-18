from __future__ import print_function

import tensorflow as tf
import numpy as np
import util
import lsp_dataset

# Parameters
learning_rate = 0.0005
training_iters = 400000
batch_size = 10
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

conv1 = util.conv2d(x, 11, n_depth, 96, 4, padding='SAME', name='conv1')
lrn1 = tf.nn.lrn(conv1, name='lrn1')
pool1 = util.maxpool2d(lrn1, 2, 2, padding='VALID', name='pool1')
# output: 27 x 27 x 96


conv2 = util.conv2d(pool1, 5, 96, 256, 1, padding='SAME', name='conv2')
lrn2 = tf.nn.lrn(conv2, name='lrn2')
pool2 = util.maxpool2d(lrn2, 2, 2, padding='VALID', name='pool2')
# output: 13 x 13 x 256

conv3 = util.conv2d(pool2, 3, 256, 384, 1, padding='SAME', name='conv3')
conv4 = util.conv2d(conv3, 3, 384, 384, 1, padding='SAME', name='conv4')
conv5 = util.conv2d(conv4, 3, 384, 256, 1, padding='SAME', name='conv5')

pool3 = util.maxpool2d(conv5, 2, 2, padding='VALID', name='pool3')
# output: 6 x 6 x 256


fc1 = util.fc2d(pool3, 6 * 6 * 256, 4096, name='fc1')
fc2 = util.fc2d(fc1, 4096, 4096, name='fc2')
# fc3 = util.softmax2d(fc2, 4096, n_classes, name='fc3')
pred = util.fc2d(fc2, 4096, n_classes, relu=False, name='fc3')


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

optimizer = util.get_loss(cost)

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
            summary, loss, ret_loss_mean = sess.run([merged, cost, loss_mean], feed_dict={x: test_x,
                                                                                          y: test_y})
            test_writer.add_summary(summary, step)
            print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Loss Mean= " + \
                  "{:.5f}".format(ret_loss_mean))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 256 mnist test images
    print("Testing Loss Mean:", \
          sess.run(loss_mean, feed_dict={x: test_x,
                                         y: test_y}))
