"""
Defense submission for team Stanford & Suns for NIPS 2017 Adversarial Examples Competition
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import csv
import pickle

import numpy as np
from scipy.misc import imread
from scipy.stats import mode
from timeit import default_timer as timer
from collections import defaultdict
import tensorflow as tf

from PIL import Image
import StringIO

slim = tf.contrib.slim

tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'checkpoint_path', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'input_dir', '', 'Input directory with images.')

tf.flags.DEFINE_string(
    'output_file', '', 'Output file to save labels.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 16, 'How many images process at one time.')

# Models Indexing
# 2 = models.InceptionResNetV2Model
# 3 = models.InceptionV3Model
# 4 = models.InceptionV4Model
# 5 = models.ResNetV1Model
# 6 = models.ResNetV2Model
# 7 = models.EnsAdvInceptionResNetV2Model
# 8 = models.AdvInceptionV3Model
# 9 = models.Ens3AdvInceptionV3Model
# 10 = models.Ens4AdvInceptionV3Model
# 11 = models.KerasXceptionModel
# 12 = models.KerasRandXceptionModel(0)
# ... 35 = models.KerasRandXceptionModel(23)
# 36..71 = Same models with jpg preprocessing

tf.flags.DEFINE_string(
    'aux_output_file', '', 'Auxiliary output file.')
tf.flags.DEFINE_string(
  'vote_models', '11,12,13,14,15,16,17,18,19', '')
tf.flags.DEFINE_string(
  'thresh_models', '67,68,69,70,71', '')
tf.flags.DEFINE_string(
  'hard_models', '7,8,9,10,43,44,45,46', '')
tf.flags.DEFINE_float(
  'back_off_thresh', 0.73, '')
tf.flags.DEFINE_integer(
  'num_wrong', 0, '')
tf.flags.DEFINE_float(
  'wt_pow', 1.0, '')
tf.flags.DEFINE_string(
  'wt_mod', '-1', '')

FLAGS = tf.flags.FLAGS
NUM_CLASSES = 1001

class Evaluator(object):
  def __init__(self, models, keras_models, keras_models2, x_input, keras_input_str):
    with tf.variable_scope('evaluator'):
      self.models = models
    with tf.variable_scope('keras-eval'):
      self.keras_models = keras_models
    with tf.variable_scope('keras-eval2'):
      self.keras_models2 = keras_models2
    self.x_input = x_input
    self.keras_input_str = keras_input_str

    self.pred_labels = [tf.argmax(m.logits, axis=1) if m else None for m in models]
    self.keras_pred_labels = [tf.argmax(m.logits, axis=1) + 1 if m else None for m in keras_models]
    self.keras_pred_labels2 = [tf.argmax(m.logits, axis=1) + 1 if m else None for m in keras_models2]

    # stats
    self.batch_num = 0
    self.image_num = 0
    self.total_time = 0

  def run(self, sess, x_batch, x_keras_batch, keras_sess, keras_sess2):
    start = timer()
    batch_labels = sess.run(self.pred_labels, feed_dict={self.x_input: x_batch})
    keras_labels = []
    keras_labels2 = []
    
    for pred, mod in zip(self.keras_pred_labels, self.keras_models):
      if mod:
        in_str = mod.ckpt + '/' + mod.name + '/' + self.keras_input_str
        keras_labels.append(keras_sess.run(pred, feed_dict={in_str: x_keras_batch}))
      else:
        keras_labels.append(None)
    for pred, mod in zip(self.keras_pred_labels2, self.keras_models2):
      if mod:
        in_str = mod.ckpt + '/' + mod.name + '/' + self.keras_input_str
        keras_labels2.append(keras_sess2.run(pred, feed_dict={in_str: x_keras_batch}))
      else:
        keras_labels2.append(None)

    end = timer()
    self.total_time += end - start
    self.image_num += len(x_batch)
    self.batch_num += 1
    print("Eval Batch {:2d}, processed {:4d} images in {:7.3f} sec".format(self.batch_num,
                                                                           self.image_num,
                                                                           self.total_time))
    return batch_labels, keras_labels + keras_labels2

def load_images(input_dir, batch_shape, jpg_quality=None):
  """Read png images from input directory in batches.

  Args:
    input_dir: input directory
    batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]

  Yields:
    filenames: list file names without path of each image
      Length of this list could be less than batch_size, in this case only
      first few images of the result are elements of the minibatch.
    images: array with all images from this batch
  """
  images = np.zeros(batch_shape)
  keras_images = np.zeros(batch_shape)
  filenames = []
  idx = 0
  batch_size = batch_shape[0]
  for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
    try:
      with tf.gfile.Open(filepath) as f:
        if jpg_quality:
          image = Image.open(f)
          buffer = StringIO.StringIO()
          image.save(buffer, "JPEG", quality=jpg_quality)
          image = Image.open(buffer)
          image = np.array(image.convert('RGB')).astype(np.float) / 255.0
        else:
          image = imread(f, mode='RGB').astype(np.float) / 255.0
    except:
      with tf.gfile.Open(filepath) as f2:
        print("Error in jpg conversion for {}, falling back to png".format(filepath))
        image = imread(f2, mode='RGB').astype(np.float) / 255.0
        print("Successfully backed off")
    # Images for inception classifier are normalized to be in [-1, 1] interval.
    images[idx, :, :, :] = image * 2.0 - 1.0
    kimage = image * 2.0 - 1.0
    keras_images[idx, :, :, :] = kimage

    filenames.append(os.path.basename(filepath))
    idx += 1
    if idx == batch_size:
      yield filenames, images, keras_images
      filenames = []
      images = np.zeros(batch_shape)
      keras_images = np.zeros(batch_shape)
      idx = 0
  if idx > 0:
    yield filenames, images, keras_images

def score_by_agree_set(hard_idxs, thresh_idxs, thresh, labels):
  thresh_maj_labs = []
  thresh_cnts = []
  score = np.repeat(0.1, len(hard_idxs))
  set_size = 0
  for labs in labels:
    thresh_labs = [labs[idx] for idx in thresh_idxs]
    thresh_maj_lab = get_wt_vote(thresh_labs, np.repeat(1.0, len(thresh_labs)))
    thresh_maj_labs.append(thresh_maj_lab)
    thresh_cnts.append(thresh_labs.count(thresh_maj_lab))

    if thresh_cnts[-1] >= thresh:
      set_size += 1
      for idx, all_idx in enumerate(hard_idxs):
        if labs[all_idx] == thresh_maj_lab:
          score[idx] += 1
  return score, set_size

def get_labs_submit(labels, simple_idxs, hard_idxs, back_off_thresh=0.73,
                    thresh=4, thresh_idxs=[42,43,44,45], wt_pow=1.0,
                    wt_mod=None, med_thresh=0.75, simple_thresh_idxs=[43,44,45,46]):
  maj_labels = []
  maj_cnts = []
  simple_score = np.zeros(len(simple_idxs))
  for labs in labels:
    simple_labs = [labs[idx] for idx in simple_idxs]
    maj_lab = get_wt_vote(simple_labs, np.repeat(1.0, len(simple_labs)))
    maj_labels.append(maj_lab)
    maj_cnts.append(simple_labs.count(maj_lab))
    for idx, all_idx in enumerate(simple_idxs):
      if labs[all_idx] == maj_lab:
        simple_score[idx] += 1
  simple_score = simple_score / (len(labels) + 0.0)

  if np.median(simple_score) > back_off_thresh:
    print("Scores", simple_score)
    if simple_score[0] > med_thresh:
      wts = np.array([2.9, 1.0, 1.0, 1.0])
    else:
      agg_size = 0
      thresh_try = len(simple_thresh_idxs)
      while agg_size < 60 and thresh_try >= 3:
        score, agg_size = score_by_agree_set(simple_idxs, simple_thresh_idxs, thresh_try, labels)
        print("Simple Thresh: {} Agg: {}".format(thresh_try, agg_size))
        thresh_try -= 1
      if agg_size >= 60:
        wts = score / max(score)
        wts = modify_wts(wts, wt_pow, wt_mod)
      else:
        wts = simple_score / max(simple_score)
        wts = modify_wts(wts, wt_pow, wt_mod)

    ret = []
    for labs in labels:
      simple_labs = [labs[idx] for idx in simple_idxs]
      ret.append(get_wt_vote(simple_labs, wts))
    return ret, wts
  else:
    agg_size = 0
    thresh_try = thresh
    while agg_size < 60 and thresh_try >= 3:
      hard_score, agg_size = score_by_agree_set(hard_idxs, thresh_idxs, thresh_try, labels)
      print("Thresh: {} Agg: {}".format(thresh_try, agg_size))
      thresh_try -= 1
    if agg_size >= 60:
      wts = hard_score / max(hard_score)
      wts = modify_wts(wts, wt_pow, wt_mod)
    else:
      wts = np.array([0, 0, 0, 0, 2.9, 1.0, 1.0, 1.0])
    ret = []
    for labs in labels:
      hard_labs = [labs[idx] for idx in hard_idxs]
      ret.append(get_wt_vote(hard_labs, wts))
    return ret, wts

def get_wt_vote(labs, wts):
  votes = np.zeros(1001)
  for model_num, mlab in enumerate(labs):
    votes[mlab] += wts[model_num]
  return np.argmax(votes)

def modify_wts(wts, wt_pow, wt_mod):
  ret = wts
  for cnt, wt in enumerate(wts):
    if wt_mod is not None:
      ret[cnt] = wts[cnt] * wt_mod[cnt]
    ret[cnt] = wts[cnt] ** wt_pow
  return ret

def main(_):
  vote_models = [int(x) for x in FLAGS.vote_models.split(',')]
  hard_models = [int(x) for x in FLAGS.hard_models.split(',')]
  thresh_models = [int(x) for x in FLAGS.thresh_models.split(',')]
  if FLAGS.wt_mod != '-1':
    wt_mods = [float(x) for x in FLAGS.wt_mod.split(',')]
  else:
    wt_mods = None

  active_models = set(vote_models) | set(hard_models) | set(thresh_models)

  batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
  num_classes = 1001

  slim_graph = tf.Graph()
  keras_graph = tf.Graph()
  keras_graph2 = tf.Graph()
  sess2 = tf.Session(graph=keras_graph)
  sess3 = tf.Session(graph=keras_graph2)
  with slim_graph.as_default():
    x_input = tf.placeholder(tf.float32, shape=batch_shape)

    import models

    initialized_vars = set()
    savers = []

    # list of models in our ensemble
    all_models = [models.InceptionResNetV2Model, models.InceptionV3Model, models.InceptionV4Model,
                  models.ResNetV1Model, models.ResNetV2Model,
                  models.EnsAdvInceptionResNetV2Model, models.AdvInceptionV3Model,
                  models.Ens3AdvInceptionV3Model, models.Ens4AdvInceptionV3Model]

    # build all the models and specify the saver
    for i, model in enumerate(all_models):
      all_models[i] = model(num_classes)
      all_models[i](x_input, FLAGS.batch_size)
      all_vars = slim.get_model_variables()
      uninitialized_vars = set(all_vars) - initialized_vars
      saver_dict = {v.op.name[len(all_models[i].ckpt) + 1:]: v for v in uninitialized_vars}
      savers.append(tf.train.Saver(saver_dict))
      initialized_vars = set(all_vars)
  
  rand_start = timer()
  num_xception_models_loaded = 0
  with keras_graph.as_default():
    keras_xception = models.KerasXceptionModel(num_classes)
    keras_xception(sess2, ckpt_path=FLAGS.checkpoint_path)
    
    rand_xceptions = []
    for idx in range(15):
      if idx + 12 in active_models or idx + 12 + 36 in active_models:
        rand_xceptions.append(models.KerasRandXceptionModel(num_classes, idx))
        rand_xceptions[-1](sess2, ckpt_path=FLAGS.checkpoint_path)
        num_xception_models_loaded += 1
      else:
        rand_xceptions.append(None)

  with keras_graph2.as_default():
    rand_xceptions2 = []
    for idx in range(15, 24):
      if idx + 12 in active_models or idx + 12 + 36 in active_models:
        rand_xceptions2.append(models.KerasRandXceptionModel(num_classes, idx))
        rand_xceptions2[-1](sess3, ckpt_path=FLAGS.checkpoint_path)
        num_xception_models_loaded += 1
      else:
        rand_xceptions2.append(None)

  rand_end = timer()
  print("Loaded {} xception models in {:7.3f} sec".format(num_xception_models_loaded, 
                                                          rand_end - rand_start))

  keras_models = [keras_xception] + rand_xceptions
  keras_models2 = rand_xceptions2

# 0 = models.InceptionResNetV2Model
# 1 = models.InceptionV3Model
# 2 = models.InceptionV4Model
# 3 = models.ResNetV1Model
# 4 = models.ResNetV2Model
# 5 = models.EnsAdvInceptionResNetV2Model
# 6 = models.AdvInceptionV3Model
# 7 = models.Ens3AdvInceptionV3Model
# 8 = models.Ens4AdvInceptionV3Model
# Keras:
# 0 = models.KerasXceptionModel
# 1 = models.KerasRandXceptionModel(0)
# .. 24 = models.KerasRandXceptionModel(23)
#

  ens_models_idxs = [0, 1, 2, 3, 4, 5, 6, 7, 8]
  ens_keras_idxs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
  ens_keras_idxs2 = [16, 17, 18, 19, 20, 21, 22, 23, 24]

  ens_models = [all_models[idx] for idx in ens_models_idxs]
  ens_keras_models = [keras_models[idx] for idx in ens_keras_idxs]
  ens_keras_models2 = [keras_models2[idx - 16] for idx in ens_keras_idxs2]

  with tf.variable_scope('evaluator'):
    evaluator = Evaluator(models=all_models, keras_models=keras_models,
                          keras_models2=keras_models2, x_input=x_input,
                          keras_input_str='input_img:0')

  all_labels = defaultdict(list)
  filenames_lst = []

  with tf.Session(graph=slim_graph) as sess:
    sess.run(tf.global_variables_initializer())
    rest_start = timer()
    for model, saver in zip(all_models, savers):
      saver.restore(sess, FLAGS.checkpoint_path + '/' + model.ckpt)
    rest_end = timer()
    print("Loaded {} models in {:7.3f} sec, beginning evaluation".format(len(all_models), rest_end - rest_start))

    first_pass = True
    for jpg_quality in [None, 15]:
      for filenames, images, keras_images in load_images(FLAGS.input_dir, batch_shape, jpg_quality):
        model_batch_labels, keras_batch_labels = evaluator.run(sess, images, keras_images, sess2, sess3)
        for idx, filename in enumerate(filenames):
          if first_pass:
            filenames_lst.append(filename)
          all_labels[filename].extend([None, None])
          all_labels[filename].extend([labs[idx] if labs is not None else None for labs in model_batch_labels])
          all_labels[filename].extend([labs[idx] if labs is not None else None for labs in keras_batch_labels])
        print("Finished evaluator")
      first_pass = False


  submit_labels = dict()
  all_labels_lst = []
  for filename in filenames_lst:
    all_labels_lst.append(all_labels[filename])
  submit_labels_lst = []

  batch_size = 100
  idx = 0
  submit_wt_avg = np.zeros(len(vote_models))
  while idx < len(all_labels_lst):
    next_idx = idx + batch_size
    batch_labels = all_labels_lst[idx:next_idx]

    simple_models = vote_models
    smart_thresh = len(thresh_models) - FLAGS.num_wrong
    submit_batch, submit_wts = get_labs_submit(batch_labels, simple_models, hard_models,
                                               back_off_thresh=FLAGS.back_off_thresh, thresh=smart_thresh,
                                               thresh_idxs=thresh_models, wt_pow=FLAGS.wt_pow)
    submit_labels_lst.extend(submit_batch)
    if len(submit_wts) != len(submit_wt_avg):
      submit_wt_avg = submit_wts
    else:
      submit_wt_avg = submit_wt_avg + submit_wts

    idx = next_idx

  submit_wt_avg = submit_wt_avg / 10.0
  print("SubmitWts: {}".format(', '.join(map(lambda x: "{:.3f}".format(x), submit_wt_avg))))
  
  for filename, label in zip(filenames_lst, submit_labels_lst):
    submit_labels[filename] = label

  with tf.gfile.Open(FLAGS.output_file, 'w') as out_file:
    for filename, label in submit_labels.items():
      out_file.write('{0},{1}\n'.format(filename, label))
    
  if FLAGS.aux_output_file != '':
    with tf.gfile.Open(FLAGS.aux_output_file, 'w') as aux_out_file:
      for filename, labels in all_labels.items():
        submit_str = str(submit_labels[filename])
        model_str = ','.join(map(str, labels))
        aux_out_file.write('{0},{1},{2}\n'.format(filename, submit_str, model_str))

if __name__ == '__main__':
  tf.app.run()
