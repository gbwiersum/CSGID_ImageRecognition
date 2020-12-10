from functools import partial
import tensorflow as tf
from ImageRecognition.utils.Inceptionv4 import create_model

AUTOTUNE = tf.data.experimental.AUTOTUNE
classes = 1001
image_size = (512, 512)


def get_feature_description():

    # Feature descriptions as reported by MARCO:
    feature_description = {
        "image/height": tf.io.VarLenFeature(tf.int64),  # image height in pixels
        "image/width": tf.io.VarLenFeature(tf.int64),  # image width in pixels
        "image/colorspace": tf.io.VarLenFeature(tf.string),  # specifying the colorspace, always 'RGB'
        "image/channels": tf.io.VarLenFeature(tf.int64),  # specifying the number of channels, always 3
        "image/class/label": tf.io.VarLenFeature(tf.int64),  # specifying the index in a normalized classification layer
        "image/class/raw": tf.io.VarLenFeature(tf.int64),
        # specifying the index in the raw (original) classification layer
        "image/class/source": tf.io.VarLenFeature(tf.int64),
        # specifying the index of the source (creator of the image)
        "image/class/text": tf.io.VarLenFeature(tf.string),
        # specifying the human-readable version of the normalized label
        "image/format": tf.io.VarLenFeature(tf.string),  # specifying the format, always 'JPEG'
        "image/filename": tf.io.VarLenFeature(tf.string),  # containing the basename of the image file
        "image/id": tf.io.VarLenFeature(tf.int64),  # specifying the unique id for the image
        "image/encoded": tf.io.VarLenFeature(tf.string),  # containing JPEG encoded image in RGB colorspace
    }
    return feature_description


def decode_image(image, im_size):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize_with_crop_or_pad(image, 512, 512)
    image = tf.image.resize(image, size=[im_size[0], im_size[1]])
    return image


def _read_tfrecord(example, labeled=True):
    tfrecord_format = (
        {
            "image/encoded": tf.io.FixedLenFeature([], tf.string),
            "image/class/label": tf.io.FixedLenFeature([], tf.int64),
        }
        if labeled
        else {"image/encoded": tf.io.FixedLenFeature([], tf.string), }
    )
    example = tf.io.parse_single_example(example, tfrecord_format)
    image = decode_image(example["image/encoded"], image_size)
    if labeled:
        label = tf.cast(example["image/class/label"], tf.int32)
        label = tf.one_hot(label, depth=4)
        return image, label
    else:
        return image


def _load_dataset(filenames, labeled=True):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False  # disable order, increase speed
    dataset = tf.data.TFRecordDataset(
        filenames
    )  # automatically interleaves reads from multiple files
    dataset = dataset.with_options(
        ignore_order
    )  # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(
        partial(_read_tfrecord, labeled=labeled), num_parallel_calls=AUTOTUNE
    )
    # returns a dataset of (image, label) pairs if labeled=True or just images if labeled=False
    return dataset


def get_dataset(filenames, im_size, labeled=True, batch_size=10):
    global image_size
    image_size = im_size
    dataset = _load_dataset(filenames, labeled=labeled)
    dataset = dataset.shuffle(2048)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.batch(batch_size)
    return dataset


def tfrecord_writer(filepath, label=None):
    image_string = open(filepath, 'rb').read()
    image_shape = tf.image.decode_jpeg(image_string).shape
    label = label
    feature = {
        "image/height": tf.int64(image_shape[0]),  # image height in pixels
        "image/width": tf.int64(image_shape[1]),  # image width in pixels
        "image/colorspace": tf.string('RGB'),  # specifying the colorspace, always 'RGB'
        "image/channels": tf.int64(3),  # specifying the number of channels, always 3
        "image/class/label": tf.int64(label),  # specifying the index in a normalized classification layer
        "image/class/raw": tf.int64(label),
        # specifying the index in the raw (original) classification layer
        "image/class/source": tf.int64(5),
        # specifying the index of the source (creator of the image) - CSGID = 5
        "image/class/text": tf.string(str(label)),
        # specifying the human-readable version of the normalized label
        "image/format": tf.string("JPEG"),  # specifying the format, always 'JPEG'
        "image/filename": tf.string(filepath),  # containing the basename of the image file
        "image/id": tf.io.VarLenFeature(tf.int64),  # specifying the unique id for the image
        "image/encoded": tf.string(image_string),  # containing JPEG encoded image in RGB colorspace

    }
    return tf.train.Example(features=tf.train.Features(feature=feature))
