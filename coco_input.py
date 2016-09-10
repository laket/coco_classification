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
NUM_CLASSES = None

def get_target_categories():
    global NUM_CLASSES
    
    with open(FLAGS.path_targets, "r") as f:
        categories = [int(l[:-1]) for l in f.readlines()]

    NUM_CLASSES = len(categories)
    print ("target {} categories".format(len(categories)))
    return categories


def enumerate_files(root_dir, categories):
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
    Returns:
      (labels, paths)
      labels = [0,1,0,1...]
    """
    labels = []
    paths = []

    for category in categories:
        cur_dir = os.path.join(root_dir, str(category))

        cur_paths = [os.path.join(cur_dir, c) for c in os.listdir(cur_dir)]
        paths += cur_paths
        labels += [category] * len(cur_paths)
    
    return labels, paths

def validate_input():
    categories = get_target_categories()
    labels, paths = enumerate_files(FLAGS.dir_val, categories)

    label, path = tf.train.slice_input_producer([labels, paths], shuffle=True, capacity=4096)
    print len(labels)
    image = read_image(path)
    preprocessed = preprocess_image(image)

    return make_batch(label, preprocessed)
    

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

def make_batch(label, image):
    labels, images = tf.train.batch([label, image],
                                    FLAGS.batch_size,
                                    num_threads=2,
                                    capacity=128
    )

    return labels, images
