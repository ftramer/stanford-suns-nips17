#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import csv
import argparse

from scipy.ndimage.interpolation import rotate
from preprocessing.vgg_preprocessing import _random_crop, _central_crop

from scipy.stats import mode
import numpy as np
from PIL import Image
import StringIO

from timeit import default_timer as timer

class DatasetMetadata(object):
  """Helper class which loads and stores dataset metadata."""

  def __init__(self, filename):
    """Initializes instance of DatasetMetadata."""
    self._true_labels = {}
    self._target_classes = {}
    with open(filename) as f:
      reader = csv.reader(f)
      header_row = next(reader)
      try:
        row_idx_image_id = header_row.index('ImageId')
        row_idx_true_label = header_row.index('TrueLabel')
        row_idx_target_class = header_row.index('TargetClass')
      except ValueError:
        raise IOError('Invalid format of dataset metadata.')
      for row in reader:
        if len(row) < len(header_row):
          # skip partial or empty lines
          continue
        try:
          image_id = row[row_idx_image_id]
          self._true_labels[image_id] = int(row[row_idx_true_label])
          self._target_classes[image_id] = int(row[row_idx_target_class])
        except (IndexError, ValueError):
          raise IOError('Invalid format of dataset metadata')

  def get_true_label(self, image_id):
    """Returns true label for image with given ID."""
    return self._true_labels[image_id]

  def get_target_class(self, image_id):
    """Returns target class for image with given ID."""
    return self._target_classes[image_id]

  def save_target_classes(self, filename):
    """Saves target classed for all dataset images into given file."""
    with open(filename, 'w') as f:
      for k, v in self._target_classes.items():
        f.write('{0}.png,{1}\n'.format(k, v))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_file')
  parser.add_argument('--metadata_file', 
                      default='/scr/yis/adver/stanford-nips17-competition/dataset/dev_dataset.csv')
  args = parser.parse_args()

  img_ids = []
  labels = []
  with open(args.input_file) as f:
    reader = csv.reader(f)
    for row in reader:
      img_ids.append(row[0].split('.')[0])
      labels.append(int(row[1]))

  data = DatasetMetadata(args.metadata_file)
  true_labels = [(data.get_true_label(img_id), img_id) for img_id in img_ids]

  total = len(true_labels)
  correct = 0
  for lbl, true_lbl in zip(labels, true_labels):
    if lbl == true_lbl[0]:
      correct += 1

  print("Score: {:.3f} ({} out of {} correct)".format(correct / float(total), correct, total))

if __name__ == '__main__':
  main()
