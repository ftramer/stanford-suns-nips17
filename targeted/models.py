import tensorflow as tf
from nets import inception, resnet_v1, resnet_v2, resnet_utils, vgg, mobilenet_v1
from preprocessing.vgg_preprocessing import preprocess_image as vgg_preprocess
from preprocessing.inception_preprocessing import preprocess_image as inception_preprocess
import numpy as np

slim = tf.contrib.slim

class KerasXceptionModel(object):
  def __init__(self, num_classes):
    self.num_classes = 1000
    self.built = False
    self.logits = None
    self.ckpt = 'keras_xception.pb'
    self.name = 'keras_xception'
    self.isKerasModel = True

  def __call__(self, sess, batch_size, image, ckpt_path=''):
    print('INFO:tensorflow:Restoring parameters from %s' % (ckpt_path + '/' + self.ckpt))
    with tf.variable_scope(self.name):
      with tf.gfile.FastGFile(ckpt_path + '/' + self.ckpt, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='keras_xception', input_map={"input_image:0": image})

    logits = sess.graph.get_tensor_by_name(self.name + '/keras_xception/output_prob:0')
    logits = tf.concat(values=[tf.ones([batch_size, 1])*(-100), logits], axis=1)
    preds = tf.argmax(logits, axis=1)
    self.image = image
    self.built = True
    self.logits = logits
    self.preds = preds
    return logits

class InceptionV3Model(object):
    """Model class for CleverHans library."""

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.built = False
        self.logits = None
        self.ckpt = 'inception_v3.ckpt'
        self.name = 'inception_v3'

    def __call__(self, x_input, batch_size=None, is_training=False):
        """Constructs model and return probabilities for given input."""
        reuse = True if self.built else None
        with slim.arg_scope(inception.inception_v3_arg_scope()):
            with tf.variable_scope(self.name):
                logits, end_points = inception.inception_v3(
                    x_input, num_classes=self.num_classes, is_training=is_training,
                    reuse=reuse)

            preds = tf.argmax(logits, axis=1)
        self.built = True
        self.logits = logits
        self.preds = preds
        return logits

class AdvInceptionV3Model(object):
    """Model class for CleverHans library."""

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.built = False
        self.logits = None
        self.ckpt = 'adv_inception_v3.ckpt'
        self.name = 'adv_inception_v3'

    def __call__(self, x_input, batch_size=None, is_training=False):
        """Constructs model and return probabilities for given input."""
        reuse = True if self.built else None
        with slim.arg_scope(inception.inception_v3_arg_scope()):
            with tf.variable_scope(self.name):
                logits, end_points = inception.inception_v3(
                    x_input, num_classes=self.num_classes, is_training=is_training,
                    reuse=reuse)

            preds = tf.argmax(logits, axis=1)
        self.built = True
        self.logits = logits
        self.preds = preds
        return logits


class Ens3AdvInceptionV3Model(object):
    """Model class for CleverHans library."""

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.built = False
        self.logits = None
        self.ckpt = 'ens3_adv_inception_v3.ckpt'
        self.name = 'ens3_adv_inception_v3'

    def __call__(self, x_input, batch_size=None, is_training=False):
        """Constructs model and return probabilities for given input."""
        reuse = True if self.built else None
        with slim.arg_scope(inception.inception_v3_arg_scope()):
            with tf.variable_scope(self.name):
                logits, end_points = inception.inception_v3(
                    x_input, num_classes=self.num_classes, is_training=is_training,
                    reuse=reuse)

            preds = tf.argmax(logits, axis=1)
        self.built = True
        self.logits = logits
        self.preds = preds
        return logits


class Ens4AdvInceptionV3Model(object):
    """Model class for CleverHans library."""

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.built = False
        self.logits = None
        self.ckpt = 'ens4_adv_inception_v3.ckpt'
        self.name = 'ens4_adv_inception_v3'

    def __call__(self, x_input, batch_size=None, is_training=False):
        """Constructs model and return probabilities for given input."""
        reuse = True if self.built else None
        with slim.arg_scope(inception.inception_v3_arg_scope()):
            with tf.variable_scope(self.name):
                logits, end_points = inception.inception_v3(
                    x_input, num_classes=self.num_classes, is_training=is_training,
                    reuse=reuse)

            preds = tf.argmax(logits, axis=1)
        self.built = True
        self.logits = logits
        self.preds = preds
        return logits


class InceptionV4Model(object):
    """Model class for CleverHans library."""

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.built = False
        self.logits = None
        self.ckpt = 'inception_v4.ckpt'
        self.name = 'inception_v4'

    def __call__(self, x_input, batch_size=None, is_training=False):
        """Constructs model and return probabilities for given input."""
        reuse = True if self.built else None
        with slim.arg_scope(inception.inception_v4_arg_scope()):
            with tf.variable_scope(self.name):
                logits, end_points = inception.inception_v4(
                    x_input, num_classes=self.num_classes, is_training=is_training,
                    reuse=reuse)

            preds = tf.argmax(logits, axis=1)
        self.built = True
        self.logits = logits
        self.preds = preds
        return logits


class InceptionResNetV2Model(object):
    """Model class for CleverHans library."""

    def __init__(self, num_classes, batch_size=None):
        self.num_classes = num_classes
        self.built = False
        self.logits = None
        self.ckpt = 'inception_resnet_v2_2016_08_30.ckpt'
        self.name = 'inception_resnet_v2'

    def __call__(self, x_input, batch_size=None, is_training=False):
        """Constructs model and return probabilities for given input."""
        reuse = True if self.built else None
        with slim.arg_scope(inception.inception_resnet_v2_arg_scope()):
            with tf.variable_scope(self.name):
                logits, end_points = inception.inception_resnet_v2(
                    x_input, num_classes=self.num_classes, is_training=is_training,
                    reuse=reuse)

            preds = tf.argmax(logits, axis=1)
        self.built = True
        self.logits = logits
        self.preds = preds
        return logits

class SmoothInceptionResNetV2Model(object):
    """Model class for CleverHans library."""

    def __init__(self, num_classes, batch_size=None):
        self.num_classes = num_classes
        self.built = False
        self.logits = None
        self.ckpt = 'inception_resnet_v2_2016_08_30.ckpt'
        self.name = 'smooth_inception_resnet_v2'

    def __call__(self, x_input, batch_size=None, is_training=False):
        """Constructs model and return probabilities for given input."""
        reuse = True if self.built else None
        with slim.arg_scope(inception.inception_resnet_v2_arg_scope()):
            with tf.variable_scope(self.name):
                min_pooled = -tf.nn.max_pool(-x_input, ksize=[1, 10, 10, 1],
                               strides=[1, 1, 1, 1], padding='SAME')
                max_pooled = tf.nn.max_pool(x_input, ksize=[1, 10, 10, 1],
                               strides=[1, 1, 1, 1], padding='SAME')
                avg_pooled = (min_pooled+max_pooled)/2
                compress_pooled = avg_pooled + (x_input-avg_pooled) * (tf.sign(max_pooled-min_pooled-0.3)+1)/2
                logits, end_points = inception.inception_resnet_v2(
                    compress_pooled, num_classes=self.num_classes, is_training=is_training,
                    reuse=reuse)

            preds = tf.argmax(logits, axis=1)
        self.built = True
        self.logits = logits
        self.preds = preds
        return logits

class EnsAdvInceptionResNetV2Model(object):
    """Model class for CleverHans library."""

    def __init__(self, num_classes, batch_size=None):
        self.num_classes = num_classes
        self.built = False
        self.logits = None
        self.ckpt = 'ens_adv_inception_resnet_v2.ckpt'
        self.name = 'ens_adv_inception_resnet_v2'

    def __call__(self, x_input, batch_size=None, is_training=False):
        """Constructs model and return probabilities for given input."""
        reuse = True if self.built else None
        with slim.arg_scope(inception.inception_resnet_v2_arg_scope()):
            with tf.variable_scope(self.name):
                logits, end_points = inception.inception_resnet_v2(
                    x_input, num_classes=self.num_classes, is_training=is_training,
                    reuse=reuse)

            preds = tf.argmax(logits, axis=1)
        self.built = True
        self.logits = logits
        self.preds = preds
        return logits


class ResNetV1Model(object):
    """Model class for CleverHans library."""

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.built = False
        self.logits = None
        self.ckpt = 'resnet_v1_50.ckpt'
        self.name = 'resnet_v1_50'

    def __call__(self, x_input, batch_size, is_training=False):
        """Constructs model and return probabilities for given input."""
        reuse = True if self.built else None

        # ResNet V1 and VGG have different preprocessing
        preproc = tf.map_fn(
            lambda img: vgg_preprocess(0.5 * 255.0 * (img + 1.0),
                                       resnet_v1.resnet_v1.default_image_size,
                                       resnet_v1.resnet_v1.default_image_size), x_input)

        with slim.arg_scope(resnet_utils.resnet_arg_scope()):
            with tf.variable_scope(self.name):
                logits, end_points = resnet_v1.resnet_v1_50(
                    preproc, num_classes=self.num_classes - 1, is_training=is_training,
                    reuse=reuse)

                # VGG and ResNetV1 don't have a background class
                background_class = tf.constant(-np.inf, dtype=tf.float32, shape=[batch_size, 1])
                logits = tf.concat([background_class, logits], axis=1)

            preds = tf.argmax(logits, axis=1)
        self.built = True
        self.logits = logits
        self.preds = preds
        return logits


class ResNetV1_152_Model(object):
    """Model class for CleverHans library."""

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.built = False
        self.logits = None
        self.ckpt = 'resnet_v1_152.ckpt'
        self.name = 'resnet_v1_152'

    def __call__(self, x_input, batch_size, is_training=False):
        """Constructs model and return probabilities for given input."""
        reuse = True if self.built else None

        # ResNet V1 and VGG have different preprocessing
        preproc = tf.map_fn(
            lambda img: vgg_preprocess(0.5 * 255.0 * (img + 1.0),
                                       resnet_v1.resnet_v1.default_image_size,
                                       resnet_v1.resnet_v1.default_image_size), x_input)

        with slim.arg_scope(resnet_utils.resnet_arg_scope()):
            with tf.variable_scope(self.name):
                logits, end_points = resnet_v1.resnet_v1_152(
                    preproc, num_classes=self.num_classes - 1, is_training=is_training,
                    reuse=reuse)

            # VGG and ResNetV1 don't have a background class
            background_class = tf.constant(-np.inf, dtype=tf.float32, shape=[batch_size, 1])
            logits = tf.concat([background_class, logits], axis=1)

            preds = tf.argmax(logits, axis=1)
        self.built = True
        self.logits = logits
        self.preds = preds
        return logits


class ResNetV2Model(object):
    """Model class for CleverHans library."""

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.built = False
        self.logits = None
        self.ckpt = 'resnet_v2_50.ckpt'
        self.name = 'resnet_v2_50'

    def __call__(self, x_input, batch_size=None, is_training=False):
        """Constructs model and return probabilities for given input."""
        reuse = True if self.built else None
        with slim.arg_scope(resnet_utils.resnet_arg_scope()):
            with tf.variable_scope(self.name):
                logits, end_points = resnet_v2.resnet_v2_50(
                    x_input, num_classes=self.num_classes, is_training=is_training,
                    reuse=reuse)

            preds = tf.argmax(logits, axis=1)
        self.built = True
        self.logits = logits
        self.preds = preds
        return logits


class ResNetV2_101_Model(object):
    """Model class for CleverHans library."""

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.built = False
        self.logits = None
        self.ckpt = 'resnet_v2_101.ckpt'
        self.name = 'resnet_v2_101'

    def __call__(self, x_input, batch_size=None, is_training=False):
        """Constructs model and return probabilities for given input."""
        reuse = True if self.built else None
        with slim.arg_scope(resnet_utils.resnet_arg_scope()):
            with tf.variable_scope(self.name):
                logits, end_points = resnet_v2.resnet_v2_101(
                    x_input, num_classes=self.num_classes, is_training=is_training,
                    reuse=reuse)

            preds = tf.argmax(logits, axis=1)
        self.built = True
        self.logits = logits
        self.preds = preds
        return logits


class ResNetV2_152_Model(object):
    """Model class for CleverHans library."""

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.built = False
        self.logits = None
        self.ckpt = 'resnet_v2_152.ckpt'
        self.name = 'resnet_v2_152'

    def __call__(self, x_input, batch_size=None, is_training=False):
        """Constructs model and return probabilities for given input."""
        reuse = True if self.built else None
        with slim.arg_scope(resnet_utils.resnet_arg_scope()):
            with tf.variable_scope(self.name):
                logits, end_points = resnet_v2.resnet_v2_152(
                    x_input, num_classes=self.num_classes, is_training=is_training,
                    reuse=reuse)

            preds = tf.argmax(logits, axis=1)
        self.built = True
        self.logits = logits
        self.preds = preds
        return logits


class VGG16(object):
    """Model class for CleverHans library."""

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.built = False
        self.logits = None
        self.ckpt = 'vgg_16.ckpt'
        self.name = 'vgg_16'

    def __call__(self, x_input, batch_size, is_training=False):
        """Constructs model and return probabilities for given input."""
        reuse = True if self.built else None

        # ResNet V1 and VGG have different preprocessing
        preproc = tf.map_fn(
            lambda img: vgg_preprocess(0.5 * 255.0 * (img + 1.0),
                                       vgg.vgg_16.default_image_size,
                                       vgg.vgg_16.default_image_size), x_input)

        with tf.variable_scope(self.name):
            logits, end_points = vgg.vgg_16(
                preproc, num_classes=self.num_classes - 1, is_training=is_training)

        # VGG and ResNetV1 don't have a background class
        background_class = tf.constant(-np.inf, dtype=tf.float32, shape=[batch_size, 1])
        logits = tf.concat([background_class, logits], axis=1)

        preds = tf.argmax(logits, axis=1)

        self.built = True
        self.logits = logits
        self.preds = preds
        return logits


class MobileNetModel(object):
    """Model class for CleverHans library."""

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.built = False
        self.logits = None
        self.ckpt = 'mobilenet_v1_1.0_224.ckpt'
        self.name = 'mobilenet_v1_1'

    def __call__(self, x_input, batch_size=None, is_training=False):
        """Constructs model and return probabilities for given input."""
        reuse = True if self.built else None

        preproc = tf.map_fn(
            lambda img: inception_preprocess(img,
                                             mobilenet_v1.mobilenet_v1.default_image_size,
                                             mobilenet_v1.mobilenet_v1.default_image_size), x_input)

        with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope()):
            with tf.variable_scope(self.name):
                logits, end_points = mobilenet_v1.mobilenet_v1(
                    preproc, num_classes=self.num_classes, is_training=is_training,
                    reuse=reuse)

            preds = tf.argmax(logits, axis=1)
        self.built = True
        self.logits = logits
        self.preds = preds
        return logits
