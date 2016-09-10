"""
handle MSCOCO's input data.

This module assumes folder tree such as
dir_out/
  1/
    xxxx.jpg
    xxxx.jpg
  2/ 
    xxxx.jpg
  directory for each label
"""

import os
import glob

import numpy as np
import tensorflow as tf
import config


FLAGS = tf.app.flags.FLAGS
_DATASET = None

def get_dataset():
    global _DATASET
    if _DATASET is None:
        _DATASET = Dataset()
    return _DATASET

class Dataset(object):
    """
    Dataset manager
    """
    def __init__(self):
        categories = self._get_target_categories()
        map_categories = {}

        #TODO remain category mapping as output file
        for idx, cat in enumerate(categories):
            map_categories[cat] = idx

        self.num_classes = len(categories)
        self.map_categories = map_categories

        self.size_validation = None

    def _get_target_categories(self):
        with open(FLAGS.path_targets, "r") as f:
            categories = [int(l[:-1]) for l in f.readlines()]

        print ("target {} categories".format(len(categories)))
        return categories


    def _enumerate_files(self, root_dir):
        """
        enuemrate data in a directory.
        Target categories are specified with args.
        root_dir/
          1/
            xxxx.jpg
          2/ (category)

        Args:
          root_dir: contains image data
          categories: list of int. categories to be enumerated
          label_mapping: dictionary from original label to this model's label, label_mapping[src] = dest
        Returns:
        (labels, paths)
        labels = [0,1,0,1...]
        """
        labels = []
        paths = []
        
        for src_category, dest_category in self.map_categories.items():
            cur_dir = os.path.join(root_dir, str(src_category))

            cur_paths = [os.path.join(cur_dir, c) for c in os.listdir(cur_dir)]
            paths += cur_paths
            labels += [dest_category] * len(cur_paths)
            
        return labels, paths

    def get_validation_size(self):
        if self.size_validation is None:
            self.validate_input()
            
        return self.size_validation

    def validate_input(self):
        categories = self._get_target_categories()
        labels, paths = self._enumerate_files(FLAGS.dir_val)

        label, path = tf.train.slice_input_producer([labels, paths], shuffle=True, capacity=4096)
        self.size_validation = len(labels)
    
        image = read_image(path)
        preprocessed = preprocess_image(image)

        return self.make_batch(label, preprocessed)
    


    def make_batch(self, label, image):
        labels, images = tf.train.batch([label, image],
                                        FLAGS.batch_size,
                                        num_threads=4,
                                        capacity=128
        )
        
        return labels, images


def read_image(path):
    content = tf.read_file(path)
    image = tf.image.decode_jpeg(content)
    return image


def preprocess_image(image):
    """
    preprocesses image tensor by such as reshape, scaling and mean substraction.
    This does NOT add noise such as flipping to use for train and test.

    :param Tensor image: tensor of image
    :return: preprocessed image
    :rtype: Tensor
    """
    batch_image = tf.expand_dims(image, 0)
    resized = tf.image.resize_images(batch_image, FLAGS.image_height, FLAGS.image_width)
    reshaped = tf.reshape(resized, [FLAGS.image_height, FLAGS.image_width, 3])

    float_image = tf.cast(reshaped, dtype=tf.float32)
    return float_image / 255
