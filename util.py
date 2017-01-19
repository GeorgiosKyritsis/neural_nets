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


def softmax2d(x, input_size, neuron_count, name='softmax'):
    with tf.name_scope(name):
        W = init_weights([input_size, neuron_count])
        b = init_biases([neuron_count])

        # Conv2D wrapper, with bias and relu activation
        if len(x.get_shape().as_list()) > 2:
            x = tf.reshape(x, [-1, input_size])
        softmax = tf.nn.xw_plus_b(x, W, b)
        return softmax


def _add_loss_summaries(total_loss):
    """Add summaries for losses in DeepPose model.

    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.

    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    return loss_averages_op


def train(total_loss):

    """Train DeepPose model.

    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.

    Args:
      total_loss: Total loss from loss().
      global_step: Integer Variable counting the number of training steps
        processed.
    Returns:
      train_op: op for training.
    """

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    var_list_all = tf.trainable_variables()
    var_list1 = var_list_all[:5]
    var_list2 = var_list_all[5:]

    learning_rate = 0.0005
    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt1 = tf.train.GradientDescentOptimizer(learning_rate*10)
        opt2 = tf.train.GradientDescentOptimizer(learning_rate)
        # grads = opt.compute_gradients(total_loss)
        # grads = tf.gradients(total_loss, var_list1 + var_list2)
        grads1 = opt1.compute_gradients(total_loss, var_list1)
        grads2 = opt2.compute_gradients(total_loss, var_list2)
        # grads1 = grads[:len(var_list1)]
        # grads2 = grads[len(var_list1):]

    # Apply gradients.
    # apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    apply_gradient_op1 = opt1.apply_gradients(grads1)
    apply_gradient_op2 = opt2.apply_gradients(grads2)
    apply_gradient_op = tf.group(apply_gradient_op1, apply_gradient_op2)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(0.999)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op


def inference(images, keep_prob=1):

    # Network Parameters
    n_input = 220  # data input (img shape: 28*28)
    n_depth = 3  # total classes (0-9 digits)
    n_classes = 28  # total classes (0-9 digits)

    conv1 = conv2d(images, 11, n_depth, 96, 4, padding='SAME', name='conv1')
    lrn1 = tf.nn.lrn(conv1, name='lrn1')
    pool1 = maxpool2d(lrn1, 2, 2, padding='VALID', name='pool1')
    # output: 27 x 27 x 96


    conv2 = conv2d(pool1, 5, 96, 256, 1, padding='SAME', name='conv2')
    lrn2 = tf.nn.lrn(conv2, name='lrn2')
    pool2 = maxpool2d(lrn2, 2, 2, padding='VALID', name='pool2')
    # output: 13 x 13 x 256

    conv3 = conv2d(pool2, 3, 256, 384, 1, padding='SAME', name='conv3')
    conv4 = conv2d(conv3, 3, 384, 384, 1, padding='SAME', name='conv4')
    conv5 = conv2d(conv4, 3, 384, 256, 1, padding='SAME', name='conv5')

    pool3 = maxpool2d(conv5, 2, 2, padding='VALID', name='pool3')
    # output: 6 x 6 x 256

    fc1 = fc2d(pool3, 6 * 6 * 256, 4096, name='fc1')
    fc2 = fc2d(fc1, 4096, 4096, name='fc2')
    # fc3 = softmax2d(fc2, 4096, n_classes, name='fc3')
    pred = fc2d(fc2, 4096, n_classes, relu=False, name='fc3')
    return pred



