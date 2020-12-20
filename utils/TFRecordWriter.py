import tensorflow as tf
import glob


# The following functions can be used to convert a value to a type compatible
# with tf.train.Example.
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def tfrecord_writer(filepath, label=5):
    image_string = open(filepath, 'rb').read()
    image_shape = tf.image.decode_jpeg(image_string).shape
    # TODO AI and Human labels need to be handled the same(?) way but we need to be able to specify.
    label = label
    feature = {
        "image/height": _int64_feature(image_shape[0]),  # image height in pixels
        "image/width": _int64_feature(image_shape[1]),  # image width in pixels
        "image/colorspace": _bytes_feature(bytes('RGB', encoding='utf8')),  # specifying the colorspace, always 'RGB'
        "image/channels": _int64_feature(3),  # specifying the number of channels, always 3
        "image/class/human": _int64_feature(label),  # specifying the index in a normalized classification layer
        "image/class/AI": _int64_feature(label),
        # specifying the index in the raw (original) classification layer
        "image/class/source": _int64_feature(5),
        # specifying the index of the source (creator of the image) - CSGID = 5
        "image/class/text": _bytes_feature(bytes(label)),
        # specifying the human-readable version of the normalized label
        "image/format": _bytes_feature(bytes("JPEG", encoding='utf8')),  # specifying the format, always 'JPEG'
        "image/filename": _bytes_feature(bytes(filepath, encoding='utf8')),  # containing the basename of the image file
        "image/id": _bytes_feature(bytes(filepath, encoding='utf8')),  # specifying the unique id for the image
        "image/encoded": _bytes_feature(image_string),  # containing JPEG encoded image in RGB colorspace
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def plate_to_tfrecord(folder):
    images = glob.glob(folder)
    wells = [s for s in images if "_E0_" in s]
    plate = dict(zip(range(1, len(wells)), wells))
    with tf.io.TFRecordWriter(folder + 'plate_data.tfrecord') as writer:
        for label, filename in plate.items():
            tf_example = tfrecord_writer(filename, label)
            writer.write(tf_example.SerializeToString())
    return str(folder + 'plate_data.tfrecord')

