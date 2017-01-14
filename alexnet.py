import tensorflow as tf
import tflearn.datasets.oxflower17 as oxflower17
import numpy as np

size_input = 227
depth_input = 3
size_output = 17


def init_weights(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def init_biases(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def generate_conv2d(layer_input, kernel_size, input_depth, output_depth, stride_size, name):
    # weights shape: [filter_size, filter_size, input_depth, output_depth]
    # biases shape: [output_depth, 1]
    # stride shape: [1, filter_size, filter_size, 1]
    # padding shape: [1, filter_size, filter_size, 1]
    weights = init_weights([kernel_size, kernel_size, input_depth, output_depth])
    biases = init_biases([output_depth])
    conv = tf.nn.conv2d(layer_input, weights, strides=[1, stride_size, stride_size, 1], padding='SAME')
    conv = tf.nn.bias_add(conv, bias=biases)
    conv = tf.nn.relu(conv, name=name)
    return conv


def generate_max_pooling(layer_input, kernel_size, stride_size, name):
    # stride shape: [1, filter_size, filter_size, 1]
    # padding shape: [1, filter_size, filter_size, 1]
    pool = tf.nn.max_pool(layer_input, ksize=[1, kernel_size, kernel_size, 1], strides=[1, stride_size, stride_size, 1],
                          padding='SAME', name=name)
    return pool


def generate_lrn(layer_input, name):
    lrn = tf.nn.lrn(layer_input, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75,
                    name=name)
    return lrn


def generate_fully_connected(layer_input, input_count, neuron_count, name):
    weights = init_weights([input_count, neuron_count])
    biaes = init_biases([neuron_count])
    inference = layer_input
    if len(layer_input.get_shape().as_list()) > 2:
        inference = tf.reshape(inference, [-1, input_count])
    fc = tf.nn.bias_add(tf.matmul(inference, weights), biaes, name=name)
    return fc


def get_loss(y_pred, y_true):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_pred, y_true))
    # _EPSILON = 1e-10
    # # generate loss
    # y_pred /= tf.reduce_sum(y_pred,
    #                         reduction_indices=len(y_pred.get_shape()) - 1,
    #                         keep_dims=True)
    # y_pred = tf.clip_by_value(y_pred, tf.cast(_EPSILON, dtype=tf.float32),
    #                           tf.cast(1. - _EPSILON, dtype=tf.float32))
    # cross_entropy = - tf.reduce_sum(y_true * tf.log(y_pred),
    #                                 reduction_indices=len(y_pred.get_shape()) - 1)
    # loss = tf.reduce_mean(cross_entropy)
    # cross_entropy = - tf.reduce_sum(y_true * tf.log(y_pred))
    # loss = tf.reduce_mean(cross_entropy)
    return loss


def alex_net():
    dropout = 0.5
    learning_rate = 0.01
    train_epoch = 10

    with tf.name_scope('input'):
        x = tf.placeholder("float", [None, size_input, size_input, 3])

    with tf.name_scope('conv1'):
        conv1 = generate_conv2d(x, 11, 3, 96, 4, 'conv1')
    with tf.name_scope('pool1'):
        pool1 = generate_max_pooling(conv1, kernel_size=3, stride_size=2, name='pool1')
    with tf.name_scope('lrn1'):
        lrn1 = generate_lrn(pool1, 'lrn1')

    with tf.name_scope('conv2'):
        conv2 = generate_conv2d(lrn1, 5, 96, 256, 1, 'conv2')
    with tf.name_scope('pool2'):
        pool2 = generate_max_pooling(conv2, 3, 2, 'pool2')
    with tf.name_scope('lrn2'):
        lrn2 = generate_lrn(pool2, 'lrn2')

    with tf.name_scope('conv3'):
        conv3 = generate_conv2d(lrn2, 3, 256, 384, 1, 'conv3')
    with tf.name_scope('conv4'):
        conv4 = generate_conv2d(conv3, 3, 384, 384, 1, 'conv4')
    with tf.name_scope('conv5'):
        conv5 = generate_conv2d(conv4, 3, 384, 256, 1, 'conv5')

    with tf.name_scope('pool3'):
        pool3 = generate_max_pooling(conv5, 3, 2, 'pool3')

    with tf.name_scope('lrn3'):
        lrn3 = generate_lrn(pool3, 'lrn3')

    with tf.name_scope('fc1'):
        fc1 = generate_fully_connected(lrn3, 8 * 8 * 256, 4096, 'fc1')
        # fc1 = tf.nn.tanh(fc1)
        fc1 = tf.nn.relu(fc1)
    with tf.name_scope('drop1'):
        drop1 = tf.nn.dropout(fc1, dropout)

    with tf.name_scope('fc2'):
        fc2 = generate_fully_connected(drop1, 4096, 4096, 'fc2')
        # fc2 = tf.nn.tanh(fc2)
        fc2 = tf.nn.relu(fc2)
    with tf.name_scope('drop2'):
        drop2 = tf.nn.dropout(fc2, dropout)

    with tf.name_scope('fc4'):
        fc3 = generate_fully_connected(drop2, 4096, size_output, 'fc3')
        net = tf.nn.softmax(fc3)

    with tf.name_scope('target'):
        y = tf.placeholder(shape=[None, size_output], dtype=tf.float32, name="y")
    with tf.name_scope('loss'):
        loss = get_loss(net, y)

    train_step = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9).minimize(loss)

    X, Y = oxflower17.load_data(one_hot=True, resize_pics=(227, 227))
    # randomly chose 80% of data as train, 20% as test
    rnd_indices = np.random.rand(len(X)) < 0.80
    train_x = X[rnd_indices]
    train_y = Y[rnd_indices]
    test_x = X[~rnd_indices]
    test_y = Y[~rnd_indices]

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(net, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # Launch the graph in a session
    with tf.Session() as sess:
        # you need to initialize all variables
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('train_logs',
                                             sess.graph)
        tf.initialize_all_variables().run()

        count = 0
        for i in range(train_epoch):
            batch_size = 100
            for index in range(0, len(train_x), batch_size):
                train_batch_x = train_x[index:index + batch_size]
                train_batch_y = train_y[index: index + batch_size]
                # train_idx = np.random.randint(len(train_x), size=10)
                # train_batch_x = train_x[train_idx]
                # train_batch_y = train_y[train_idx]
                out_train, out_fc3, out_net = sess.run(
                    [train_step, fc3, net],
                    feed_dict={x: train_batch_x, y: train_batch_y})
                # print out_net
                if count % 10 == 0:
                    test_idx = np.random.randint(len(test_x), size=10)
                    # test_batch_x = test_x[test_idx]
                    # test_batch_y = test_y[test_idx]
                    test_batch_x = train_x
                    test_batch_y = train_y
                    print "test accuracy %g" % accuracy.eval(feed_dict={
                        x: test_batch_x, y: test_batch_y})
                count += 1


alex_net()

'''
# model training
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
'''
