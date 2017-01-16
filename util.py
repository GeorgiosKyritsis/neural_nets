import tensorflow as tf


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape))


def init_biases(shape):
    return tf.Variable(tf.random_normal(shape))


# Create some wrappers for simplicity
def conv2d(x, kernel_size, kernel_depth, kernel_count, strides=1, padding='VALID', name='conv2d'):
    W = init_weights([kernel_size, kernel_size, kernel_depth, kernel_count])
    b = init_biases([kernel_count])
    # Conv2D wrapper, with bias and relu activation
    with tf.name_scope(name):
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padding)
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)


def maxpool2d(x, kernel_size=2, stride=2, padding='VALID',name='maxpool'):
    # MaxPool2D wrapper
    with tf.name_scope(name):
        return tf.nn.max_pool(x, ksize=[1, kernel_size, kernel_size, 1], strides=[1, stride, stride, 1],
                          padding=padding)


def fc2d(x, input_size, neuron_count, relu=True, name='fc'):
    with tf.name_scope(name):
        W = init_weights([input_size, neuron_count])
        b = init_biases([neuron_count])
        # Conv2D wrapper, with bias and relu activation
        if len(x.get_shape().as_list()) > 2:
            x = tf.reshape(x, [-1, input_size])
        fc1 = tf.add(tf.matmul(x, W), b)
        if not relu:
            return fc1
        return tf.nn.relu(fc1)
