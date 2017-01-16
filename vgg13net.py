from __future__ import print_function

import tensorflow as tf
import tflearn.datasets.oxflower17 as oxflower17
import numpy as np
import util

# Parameters
learning_rate = 0.0001
training_iters = 400000
batch_size = 300
display_step = 10

# Network Parameters
n_input = 224  # data input (img shape: 28*28)
n_depth = 3  # total classes (0-9 digits)
n_classes = 17  # total classes (0-9 digits)
dropout = 0.75  # Dropout, probability to keep units

with tf.name_scope('input'):
    # tf Graph input
    x = tf.placeholder(tf.float32, [None, n_input, n_input, n_depth])
with tf.name_scope('output'):
    y = tf.placeholder(tf.float32, [None, n_classes])

keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

# Block 1
conv1 = util.conv2d(x, 3, n_depth, 64, strides=1, padding='SAME', name='conv1')
conv2 = util.conv2d(conv1, 3, 64, 64, strides=1, padding='SAME', name='conv2')
pool1 = util.maxpool2d(conv2, 2, 2, name='maxpool1')
# Output: 112 x 112 x 64

# Block 2
conv3 = util.conv2d(pool1, 3, 64, 128, strides=1, padding='SAME', name='conv3')
conv4 = util.conv2d(conv3, 3, 128, 128, strides=1, padding='SAME', name='conv4')
pool2 = util.maxpool2d(conv4, 2, 2, name='maxpool2')
# Output: 56 x 56 x 128

# Block 3
conv5 = util.conv2d(pool2, 3, 128, 256, strides=1, padding='SAME', name='conv5')
conv6 = util.conv2d(conv5, 3, 256, 256, strides=1, padding='SAME', name='conv6')
pool3 = util.maxpool2d(conv6, 2, 2, name='maxpool3')
# Output: 28 x 28 x 256

# Block 4
conv7 = util.conv2d(pool3, 3, 256, 512, strides=1, padding='SAME', name='conv7')
conv8 = util.conv2d(conv7, 3, 512, 512, strides=1, padding='SAME', name='conv8')
pool4 = util.maxpool2d(conv8, 2, 2, name='maxpool4')
# Output: 14 x 14 x 512

# Block 5
conv9 = util.conv2d(pool4, 3, 512, 512, strides=1, padding='SAME', name='conv9')
conv10 = util.conv2d(conv9, 3, 512, 512, strides=1, padding='SAME', name='conv10')
pool5 = util.maxpool2d(conv10, 2, 2, name='maxpool5')
# Output: 7 x 7 x 512

fc1 = util.fc2d(pool5, 7 * 7 * 512, 4096, name='fc1')
fc1 = tf.nn.dropout(fc1, dropout, name='dropout1')
fc2 = util.fc2d(fc1, 4096, 4096, name='fc2')
fc2 = tf.nn.dropout(fc2, dropout, name='dropout2')
fc3 = util.fc2d(fc2, 4096, n_classes, relu=False, name='fc3')

# Construct model
pred = fc3

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
tf.summary.scalar('cost', cost)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
tf.summary.scalar('accuracy', accuracy)
merged = tf.summary.merge_all()

# Initializing the variables
init = tf.initialize_all_variables()

X, Y = oxflower17.load_data(one_hot=True, resize_pics=(224, 224))
# randomly chose 80% of data as train, 20% as test
rnd_indices = np.random.rand(len(X)) < 0.80

train_x = X[rnd_indices]
train_y = Y[rnd_indices]
test_x = X[~rnd_indices]
test_y = Y[~rnd_indices]

# Launch the graph
with tf.Session() as sess:
    train_writer = tf.train.SummaryWriter('train_logs/vgg13_net',
                                          sess.graph)
    test_writer = tf.train.SummaryWriter('test_logs/vgg13_net',
                                         sess.graph)

    def print_shape(text, t):
        print( str(text) + " " + str(t.shape))
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        train_idx = np.random.randint(len(train_x), size=batch_size)
        batch_x = train_x[train_idx]
        batch_y = train_y[train_idx]
        # Run optimization op (backprop)
        # summary, out_pool1, out_pool2, out_pool3, out_pool4, out_pool5, out_fc1, out_fc2, out_fc3, op = sess.run(
        #     [merged, pool1, pool2, pool3, pool4, pool5, fc1, fc2, fc3, optimizer], feed_dict={x: batch_x, y: batch_y,
        #                                                                                       keep_prob: dropout})
        #
        # print_shape('pool1', out_pool1)
        # print_shape('pool2', out_pool2)
        # print_shape('pool3', out_pool3)
        # print_shape('pool4', out_pool4)
        # print_shape('pool5', out_pool5)
        # print_shape('fc1', out_fc1)
        # print_shape('fc2', out_fc2)
        # print_shape('fc3', out_fc3)

        sess.run(
             optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
        # train_writer.add_summary(summary, step)
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            summary, loss, acc = sess.run([merged, cost, accuracy], feed_dict={x: test_x,
                                                                               y: test_y,
                                                                               keep_prob: 1.})
            test_writer.add_summary(summary, step)
            print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 256 mnist test images
    print("Testing Accuracy:", \
          sess.run(accuracy, feed_dict={x: test_x,
                                        y: test_y,
                                        keep_prob: 1.}))
