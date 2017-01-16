from __future__ import print_function

import tensorflow as tf
import tflearn.datasets.oxflower17 as oxflower17
import numpy as np
import util

# Parameters
learning_rate = 0.001
training_iters = 200000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 227  # data input (img shape: 28*28)
n_depth = 3  # total classes (0-9 digits)
n_classes = 17  # MNIST total classes (0-9 digits)
dropout = 0.75  # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input, n_input, n_depth])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

# Convolution Layer
conv1 = util.conv2d(x, 11, n_depth, 96, strides=4)
# Max Pooling (down-sampling)
conv1 = util.maxpool2d(conv1, 3, 2)

# Convolution Layer
conv2 = util.conv2d(conv1, 5, 96, 256, 2)
# Max Pooling (down-sampling)
conv2 = util.maxpool2d(conv2, 2, 2)

# Fully connected layer

fc1 = util.fc2d(conv2, 6 * 6 * 256, 1024)

# Apply Dropout
fc1 = tf.nn.dropout(fc1, dropout)

# Output, class prediction
out = tf.add(tf.matmul(fc1, util.init_weights([1024, n_classes])), util.init_biases([n_classes]))

# Construct model
pred = out

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
tf.summary.scalar('cost', cost)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
tf.summary.scalar('accuracy', accuracy)
merged = tf.summary.merge_all()

# Initializing the variables
init = tf.initialize_all_variables()

X, Y = oxflower17.load_data(one_hot=True, resize_pics=(227, 227))
# randomly chose 80% of data as train, 20% as test
rnd_indices = np.random.rand(len(X)) < 0.80

train_x = X[rnd_indices]
train_y = Y[rnd_indices]
test_x = X[~rnd_indices]
test_y = Y[~rnd_indices]

# Launch the graph
with tf.Session() as sess:
    train_writer = tf.train.SummaryWriter('train_logs/simple_net',
                                          sess.graph)
    test_writer = tf.train.SummaryWriter('test_logs/simple_net',
                                          sess.graph)
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        train_idx = np.random.randint(len(train_x), size=batch_size)
        batch_x = train_x[train_idx]
        batch_y = train_y[train_idx]
        # Run optimization op (backprop)
        summary, op = sess.run([merged, optimizer], feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: dropout})
        train_writer.add_summary(summary, step)
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            summary, loss, acc = sess.run([merged, cost, accuracy], feed_dict={x: test_x,
                                                              y: test_y,
                                                              keep_prob: 1.})
            print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
            test_writer.add_summary(summary, step)
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 256 mnist test images
    print("Testing Accuracy:", \
          sess.run(accuracy, feed_dict={x: test_x,
                                        y: test_y,
                                        keep_prob: 1.}))
