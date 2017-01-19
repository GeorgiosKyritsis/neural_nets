import LSPGlobals
from  LSPGlobals import FLAGS
import os
import urllib
# import urllib.request as url_request
import sys
import zipfile
import glob
import numpy as np
from scipy.io import loadmat
from scipy import misc
import tensorflow as tf
import numpy as np
import os.path


def get_dataset():

    # if os.path.isfile('LSP_data/train_x.npz'):
    #     f_train_x = open("LSP_data/train_x.npz", 'r')
    #     f_train_y = open("LSP_data/train_y.npz", 'r')
    #     f_test_x = open("LSP_data/test_x.npz", 'r')
    #     f_test_y = open("LSP_data/test_y.npz", 'r')
    #
    #     train_x = np.load(f_train_x)
    #     train_y = np.load(f_train_y)
    #     test_x = np.load(f_test_x)
    #     test_y = np.load(f_test_y)
    #     f_train_x.close()
    #     f_test_x.close()
    #     f_train_y.close()
    #     f_test_y.close()
    #     return train_x, train_y, test_x, test_y

    filename = maybe_download()

    if not os.path.exists(os.path.join(FLAGS.data_dir, FLAGS.extracted_file)):
        extract_file(filename)
    else:
        print('Already Extracted.')

    train_x, train_y, test_x, test_y = parse_resize_image_and_labels()
    f_train_x = open("LSP_data/train_x.npz", 'w')
    f_train_y = open("LSP_data/train_y.npz", 'w')
    f_test_x = open("LSP_data/test_x.npz", 'w')
    f_test_y = open("LSP_data/test_y.npz", 'w')

    np.save(f_train_x, f_train_x)
    np.save(f_train_y, f_train_y)
    np.save(f_test_x, f_test_x)
    np.save(f_test_y, f_test_y)
    f_train_x.close()
    f_test_x.close()
    f_train_y.close()
    f_test_y.close()
    return train_x, train_y, test_x, test_y


def maybe_download():
    if not os.path.exists(FLAGS.data_dir):
        os.mkdir(FLAGS.data_dir)
    file_path = os.path.join(FLAGS.data_dir, FLAGS.comp_filename)
    if not os.path.exists(file_path):
        print('Downloading ', file_path, '.')
        file_path, _ = urllib.urlretrieve(FLAGS.download_url + FLAGS.comp_filename, file_path)
        stat_info = os.stat(file_path)
        print('Successfully downloaded', stat_info.st_size, 'bytes.')
    else:
        print(file_path, 'already exists.')

    return file_path


def download_progress(count, block_size, total_size):
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r Downloaded %d%% of %d megabytes" % (percent, total_size / (1024 * 1024)))
    sys.stdout.flush()


def extract_file(filename):
    print('Extracting ', filename, '.')
    opener, mode = zipfile.ZipFile, 'r'
    # cwd = os.getcwd()
    # os.chdir(os.path.dirname(filename))
    try:
        zip_file = opener(filename, mode)
        try: zip_file.extractall(path=FLAGS.data_dir)
        finally: zip_file.close()
    finally:
        # os.chdir(cwd)
        print('Done extracting')


def parse_resize_image_and_labels():
    print('Resizing and packing images and labels to bin files.\n')
    np.random.seed(1701)  # to fix test set

    jnt_fn = FLAGS.data_dir + 'joints.mat'

    joints = loadmat(jnt_fn)
    joints = joints['joints'].swapaxes(0, 2).swapaxes(1, 2)
    invisible_joints = joints[:, :, 2] < 0.5
    joints[invisible_joints] = 0
    joints = joints[..., :2]

    image_list = np.asarray(sorted(glob.glob(FLAGS.orimage_dir + '*.jpg')))

    image_indexes = list(range(0, len(image_list)))
    np.random.shuffle(image_indexes)

    train_validation_split = int(len(image_list)*FLAGS.train_set_ratio)
    validation_test_split = int(len(image_list)*(FLAGS.train_set_ratio+FLAGS.validation_set_ratio))

    train_indexes = np.asarray(image_indexes[:train_validation_split])
    validation_indexes = np.asarray(image_indexes[train_validation_split:validation_test_split])
    test_indexes = np.asarray(image_indexes[validation_test_split:])

    train_x , train_y = generate_data(image_list[train_indexes], joints[train_indexes])
    test_x , test_y = generate_data(image_list[test_indexes], joints[test_indexes])
    # write_to_tfrecords(train_x, train_y, 'train')
    # write_to_tfrecords(image_list[validation_indexes], joints[validation_indexes], 'validation')
    # write_to_tfrecords(test_x, test_y, 'test')

    print('Done.')
    return train_x, train_y, test_x, test_y


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_feature_list(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def prepare_image(original_image_path):
    image = misc.imread(original_image_path)
    scaled_image = misc.imresize(image, (FLAGS.input_size, FLAGS.input_size), interp='bicubic')
    return scaled_image, image.shape[0], image.shape[1]


def scale_label(label, original_height, original_width):
    label[:, 0] *= (FLAGS.input_size / float(original_width))
    label[:, 1] *= (FLAGS.input_size / float(original_height))
    return label.reshape(LSPGlobals.TotalLabels)


def generate_data(image_paths, labels):
    num_examples = image_paths.shape[0]
    x = np.array([])
    y = np.array([])
    # for index in range(num_examples):
    for index in range(min(10, num_examples)):
        image, or_height, or_width = prepare_image(image_paths[index])  # FIXME read file
        label = scale_label(labels[index], or_height, or_width)
        if index == 0:
            x = np.array([image])
            y = np.array([label])
        else:
            x = np.concatenate((x, [image]), axis=0)
            y = np.concatenate((y, [label]), axis=0)
    return x, y


def write_to_tfrecords(image_paths, labels, name):
    num_examples = image_paths.shape[0]

    filename = os.path.join(FLAGS.data_dir, name + '.tfrecords')

    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)

    for index in range(num_examples):
        image, or_height, or_width = prepare_image(image_paths[index])  # FIXME read file
        image_raw = image.tostring()

        label = scale_label(labels[index], or_height, or_width)

        features = tf.train.Features(feature={
            # 'height': _int64_feature(FLAGS.input_size),
            # 'width': _int64_feature(FLAGS.input_size),
            # 'depth': _int64_feature(FLAGS.input_depth),
            'label': _int64_feature_list(label.astype(int).tolist()),
            'image_raw': _bytes_feature(image_raw)})

        example = tf.train.Example(features=features)

        writer.write(example.SerializeToString())

        print 'progress ' + str(index) + '/' + str(num_examples)

    writer.close()


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)
    # The serialized example is converted back to actual values.
    # One needs to describe the format of the objects to be returned
    features = tf.parse_single_example(
        serialized_example,
        features={
            # We know the length of both fields. If not the
            # tf.VarLenFeature could be used
            'label': tf.FixedLenFeature([LSPGlobals.TotalLabels], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string)
        })

    # now return the converted data
    image_as_vector = tf.decode_raw(features['image_raw'], tf.uint8)
    image_as_vector.set_shape([LSPGlobals.TotalImageBytes])
    image = tf.reshape(image_as_vector, [FLAGS.input_size, FLAGS.input_size, FLAGS.input_depth])
    # Convert from [0, 255] -> [-0.5, 0.5] floats.
    image_float = tf.cast(image, tf.float32) * (1. / 255) - 0.5

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    label = tf.cast(features['label'], tf.int32)

    return label, image_float


def inputs(is_train):
    train_set_file = os.path.join(FLAGS.data_dir, 'train.tfrecords')
    validation_set_file = os.path.join(FLAGS.data_dir, 'validation.tfrecords')
    """Reads input data num_epochs times."""
    filename = train_set_file if is_train else validation_set_file

    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
            [filename], num_epochs=None)

        # get single examples
        label, image = read_and_decode(filename_queue)

        # groups examples into batches randomly
        images_batch, labels_batch = tf.train.shuffle_batch(
            [image, label], batch_size=FLAGS.batch_size,
            capacity=3000,
            min_after_dequeue=1000)

        return images_batch, labels_batch




if __name__ == "__main__":
    get_dataset()










