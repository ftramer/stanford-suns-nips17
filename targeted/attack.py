"""Implementation of sample attack."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import numpy as np
import csv
import pandas as pd
from PIL import Image
import StringIO
import tensorflow as tf
from timeit import default_timer as timer

slim = tf.contrib.slim

tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'checkpoint_path', './model_ckpts', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'input_dir', '../../dataset/images', 'Input directory with images.')

tf.flags.DEFINE_string(
    'output_dir', './output_dir', 'Output directory with images.')

tf.flags.DEFINE_float(
    'max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 10, 'How many images process at one time.')

tf.flags.DEFINE_integer(
    'iternum', 8, 'How many iterations does the attacker runs.')

tf.flags.DEFINE_float(
    'learning_rate', 0.2, 'The learning rate of attacker.')

tf.flags.DEFINE_float(
    'margin', 0.01, 'margin parameter in the loss function.')

tf.flags.DEFINE_string(
    'blackbox_train', '0,1,2,3,4,10', 'models for blackbox training.')

tf.flags.DEFINE_string(
    'whitebox_train', '5,6,7,8,9', 'models for whitebox training.')

tf.flags.DEFINE_string(
    'test', '0,1,2,3,4,5,6,7,8,9,10', 'models for testing.')

FLAGS = tf.flags.FLAGS

def string_to_list(s):
    return [int(x) for x in filter(None, s.split(','))]

def compress_by_jpeg(images):
    jpeg_images = np.zeros(images.shape)
    for i in range(images.shape[0]):
        buffer = StringIO.StringIO()
        raw = (((images[i, :, :, :] + 1.0) * 0.5) * 255.0).astype(np.uint8)
        Image.fromarray(raw).save(buffer, "JPEG", quality=15)
        jpeg_images[i,:,:,:] = np.array(Image.open(buffer).convert('RGB')).astype(np.float) / 255.0
    return jpeg_images

def load_target_class(input_dir):
    """Loads target classes."""
    with tf.gfile.Open(os.path.join(input_dir, 'target_class.csv')) as f:
        return {row[0]: int(row[1]) for row in csv.reader(f) if len(row) >= 2}

def load_images(input_dir, batch_shape):
    images = np.zeros(batch_shape)
    filenames = []

    target_class_dict = load_target_class(input_dir)
    target_class_batch = []

    idx = 0
    batch_size = batch_shape[0]
    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
        with tf.gfile.Open(filepath) as f:
            # original images
            image = np.array(Image.open(f).convert('RGB')).astype(np.float) / 255.0

            # JPEG images
            #buffer = StringIO.StringIO()
            #Image.open(f).save(buffer, "JPEG", quality=15)
            #jpeg_image = np.array(Image.open(buffer).convert('RGB')).astype(np.float) / 255.0

        # Images for inception classifier are normalized to be in [-1, 1] interval.
        images[idx, :, :, :] = image * 2.0 - 1.0
        filenames.append(os.path.basename(filepath))

        # Get target class.
        fname = os.path.basename(filepath)
        target_class = target_class_dict[fname]
        target_class_batch.append(target_class)

        idx += 1
        if idx == batch_size:
            yield filenames, images, target_class_batch
            filenames = []
            target_class_batch = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images, target_class_batch


def save_images(images, filenames, output_dir):
    for i, filename in enumerate(filenames):
        # Images for inception classifier are normalized to be in [-1, 1] interval,
        # so rescale them back to [0, 1].
        with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
            img = (((images[i, :, :, :] + 1.0) * 0.5) * 255.0).astype(np.uint8)
            Image.fromarray(img).save(f, format='PNG')


class Evaluator(object):
    def __init__(self, name, models, image, image_input, true_label, test):
        errors = []
        for i in test:
            correct_prediction = tf.equal(tf.argmax(models[i].logits, axis=1), tf.cast(true_label, tf.int64))
            error = 1 - tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            errors.append(error)
        self.name = name
        self.errors = errors
        self.processed_batch_num = 0
        self.overall_errors = np.zeros(len(test))
        self.label = true_label
        self.image_input = image_input
        self.assign_image = tf.assign(image, image_input)

    def run(self, sess, image_input, y):
        sess.run(self.assign_image, feed_dict={self.image_input: image_input})
        errors = sess.run(self.errors, feed_dict={self.label: y})
        print('%s evaluation errors: %s' % (self.name, errors))

        self.processed_batch_num += 1
        self.overall_errors += errors
        if self.processed_batch_num % 10 == 0:
            print('%s overall evaluation errors: %s' % (self.name, self.overall_errors / self.processed_batch_num))

class Attacker(object):
    def __init__(self, name, models, image_input, image, true_label, max_epsilon, k, train, test,
                 optimizer, loss_type, margin, learning_rate):
        self.name = name
        self.models = models
        self.max_epsilon = max_epsilon
        self.k = k
        self.processed_batch_num = 0
        self.overall_train_errors = np.zeros(len(train))
        self.overall_test_errors = np.zeros(len(test))
        self.optimizer = optimizer
        self.loss_type = loss_type

        # placeholders
        self.label = true_label
        self.image_input = image_input
        self.image = image
        self.assign_image = tf.assign(image, image_input)
        self.assign_add_image = tf.assign_add(image, image_input)

        label_mask = tf.one_hot(true_label, 1001, on_value=1.0, off_value=0.0, dtype=tf.float32)

        def define_errors(model_indices):
            errors = []
            for i in model_indices:
                correct_prediction = tf.equal(tf.argmax(models[i].logits, axis=1), tf.cast(true_label, tf.int64))
                error = 1 - tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                errors.append(error)
            return errors

        self.train_errors = define_errors(train)
        self.test_errors = define_errors(test)

        # define average loss
        self.average_loss = 0
        for i in train:
            self.average_loss += tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=models[i].logits, labels=true_label)) / len(train)

        # define mixture loss
        softmax_prob_sum = 0
        for i in train:
            softmax_prob_sum += tf.reduce_sum(tf.nn.softmax(models[i].logits) * label_mask, axis=1)
        self.mixture_loss = (-1.0) * tf.reduce_mean(tf.log(margin + softmax_prob_sum))

        # define hinge loss
        hinge_losses = []
        for i in train:
            label_logits = tf.reduce_sum(models[i].logits * label_mask, axis=1)
            max_logits = tf.reduce_max(models[i].logits - label_mask * 100, axis=1)
            hinge_losses.append(label_logits - max_logits)
        self.hinge_loss = tf.reduce_mean(tf.reduce_max(tf.stack(hinge_losses, axis=0), axis=0))

        # define gradient
        grad = None
        if loss_type == 'average':
            grad = tf.gradients(self.average_loss, image)[0]
        if loss_type == 'mixture':
            grad = tf.gradients(self.mixture_loss, image)[0]
        if loss_type == 'hinge':
            grad = tf.gradients(self.hinge_loss, image)[0]

        # define optimization step
        opt = None
        if optimizer == 'sgd':
            opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate * max_epsilon)
            self.all_model_gradient_step = opt.apply_gradients([(tf.sign(grad), image)])
        if optimizer == 'rmsprop':
            opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate * max_epsilon, epsilon=1e-9, decay=0.9)
            self.all_model_gradient_step = opt.apply_gradients([(grad, image)])
        self.apply_null = opt.apply_gradients([(tf.zeros(image.get_shape().as_list(), dtype=tf.float32), image)])

        # define clipping step
        clipped_image = tf.clip_by_value(image, image_input - max_epsilon, image_input + max_epsilon)
        clipped_image = tf.clip_by_value(clipped_image, -1, 1)
        self.clipping_step = tf.assign(image, clipped_image)

    def run(self, sess, x_batch, y=None):
        sess.run(self.assign_image, feed_dict={self.image_input: x_batch})
        if y is None:
            y = sess.run(self.models[0].preds)

        start = timer()
        if self.optimizer == 'rmsprop':
            for _ in range(200):
                sess.run(self.apply_null)
        for i in range(self.k):
            sess.run(self.all_model_gradient_step, feed_dict={self.label: y, self.image_input: x_batch})
            sess.run(self.clipping_step, feed_dict={self.image_input: x_batch})
        end = timer()

        x_adv, train_errors_after_attack, test_errors_after_attack = sess.run(
            [self.image, self.train_errors, self.test_errors], feed_dict={self.label: y})

        print('%s -- time: %g sec, train_errors: %s, test_errors: %s' % (
            self.name, end - start, train_errors_after_attack, test_errors_after_attack))
        sys.stdout.flush()

        self.processed_batch_num += 1
        self.overall_train_errors += train_errors_after_attack
        self.overall_test_errors += test_errors_after_attack
        return y, (x_adv - x_batch)


def main(_):
    full_start = timer()
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # eps is a difference between pixels so it should be in [0, 2] interval.
    # Renormalizing epsilon from [0, 255] to [0, 2].
    eps = 2.0 * FLAGS.max_epsilon / 255.0
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    num_classes = 1001

    tf.logging.set_verbosity(tf.logging.INFO)

    with tf.Graph().as_default():
        # Prepare graph
        image_input = tf.placeholder(tf.float32, shape=batch_shape)
        image = tf.get_variable('adversarial_image', shape=batch_shape)
        label = tf.placeholder(tf.int32, shape=[FLAGS.batch_size])
        sess = tf.Session()

        import models

        initialized_vars = set()
        savers = []

        # list of models in our ensemble
        # model 0-4
        all_models = [models.InceptionResNetV2Model, models.InceptionV3Model, models.InceptionV4Model,
                      models.ResNetV1Model, models.ResNetV2Model, models.VGG16]
        # model 5-10
        all_models += [models.EnsAdvInceptionResNetV2Model, models.AdvInceptionV3Model, models.Ens3AdvInceptionV3Model,
                      models.Ens4AdvInceptionV3Model, models.KerasXceptionModel]
        # model 11
        all_models += [models.SmoothInceptionResNetV2Model]
        blackbox_train = string_to_list(FLAGS.blackbox_train)
        whitebox_train = string_to_list(FLAGS.whitebox_train)
        test = string_to_list(FLAGS.test)
        indices_to_load = [index for index in range(len(all_models)) if
                           index in blackbox_train + whitebox_train + test]

        # build all the models and specify the saver
        for i, model in enumerate(all_models):
            all_models[i] = model(num_classes)
            if hasattr(all_models[i], 'isKerasModel') and all_models[i].isKerasModel:
                if i in indices_to_load:
                    all_models[i](sess, batch_size=FLAGS.batch_size, image=image, ckpt_path=FLAGS.checkpoint_path)
                savers.append(None)
            else:
                all_models[i](image, FLAGS.batch_size)
                all_vars = slim.get_model_variables()
                uninitialized_vars = set(all_vars) - initialized_vars
                saver_dict = {v.op.name[len(all_models[i].name) + 1:]: v for v in uninitialized_vars}
                savers.append(tf.train.Saver(saver_dict))
                initialized_vars = set(all_vars)

        #whitebox_ratio = max(2.0 / FLAGS.max_epsilon, 0.2)
        #blackbox_ratio = 1 - whitebox_ratio
        whitebox_ratio = 1.0
        blackbox_ratio = 0.0
        with tf.variable_scope('whitebox-attacker'):
            with tf.variable_scope('blackbox-attacker'):
                blackbox_attacker = Attacker(name='blackbox-attacker', models=all_models, image_input=image_input,
                                             image=image, true_label=label,
                                             max_epsilon=eps * blackbox_ratio, k=FLAGS.iternum, train=blackbox_train,
                                             test=[], optimizer='rmsprop',
                                             loss_type='mixture', margin=FLAGS.margin,
                                             learning_rate=FLAGS.learning_rate)

            whitebox_attacker = Attacker(name='whitebox-attacker', models=all_models, image_input=image_input,
                                         image=image, true_label=label,
                                         max_epsilon=eps * whitebox_ratio, k=FLAGS.iternum, train=whitebox_train,
                                         test=[], optimizer='sgd',
                                         loss_type='mixture', margin=FLAGS.margin, learning_rate=FLAGS.learning_rate)

        with tf.variable_scope('raw_evaluator'):
            original_eval = Evaluator(name='original', models=all_models, image_input=image_input, image=image, true_label=label, test=test)
        with tf.variable_scope('jpeg_evaluator'):
            jpeg_eval = Evaluator(name='jpeg', models=all_models, image_input=image_input, image=image, true_label=label, test=test)

        # Run computation
        tot_time = 0.0
        processed = 0.0

        sess.run(tf.global_variables_initializer())
        for i in indices_to_load:
            if hasattr(all_models[i], 'isKerasModel') and all_models[i].isKerasModel:
                pass
            else:
                savers[i].restore(sess, FLAGS.checkpoint_path + '/' + all_models[i].ckpt)
        print("Initialization done after {} sec".format(timer() - full_start))

        for filenames, images, target_classes in load_images(FLAGS.input_dir, batch_shape):
            start = timer()
            print('batch %d' % (blackbox_attacker.processed_batch_num + 1))
            y, blackbox_perturb = blackbox_attacker.run(sess, images, target_classes)
            if whitebox_ratio > 0:
                _, whitebox_perturb = whitebox_attacker.run(sess, images + blackbox_perturb, target_classes)
            else:
                whitebox_perturb = 0
            images += blackbox_perturb + whitebox_perturb
            original_eval.run(sess, images, target_classes)
            jpeg_eval.run(sess, compress_by_jpeg(images), target_classes)

            if FLAGS.output_dir != '':
                save_images(images, filenames, FLAGS.output_dir)

            end = timer()
            tot_time += end - start
            processed += FLAGS.batch_size

        full_end = timer()
        print("DONE: Processed {} images in {} sec".format(processed, full_end - full_start))

if __name__ == '__main__':
    tf.app.run()
