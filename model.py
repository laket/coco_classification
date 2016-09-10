#!/usr/bin/python

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import gen_nn_ops

import coco_input

def _var(name, shape, wd=0.001,initializer=None):
    #sqrt(3. / (in + out))
    if initializer is None:
        initializer = tf.contrib.layers.xavier_initializer()
        
    var = tf.get_variable(name, shape, initializer=initializer)
    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

class Layer(object):
    """
    Layer with convolution and pooling
    """
    def __init__(self, name, output_ch, num_conv=1, retain_ratio=0.5, is_train=False):
        self.name = name
        self.output_ch = output_ch
        self.num_conv = num_conv
        self.retain_ratio = retain_ratio
        self.is_train = is_train

    def inference(self, in_feat):
        self.input_shape = in_feat.get_shape()
        N, H, W, C = self.input_shape
        feat = in_feat

        with tf.variable_scope(self.name):
            with tf.variable_scope("conv0"):
                self.w = _var("W", [3,3,C,self.output_ch])
                self.b = _var("b", [self.output_ch],initializer=tf.constant_initializer())
                    
                feat = tf.nn.conv2d(feat, self.w, strides=[1,1,1,1],padding="VALID")
                feat = feat + self.b
                feat = tf.nn.relu(feat)
                    
            feat = tf.nn.max_pool(feat, [1,2,2,1], strides=[1,2,2,1],padding="SAME")

            if self.is_train:
                feat = tf.nn.dropout(feat, keep_prob=self.retain_ratio)
                    
        return feat


class Network(object):
    def __init__(self, is_train):
        self.is_train = is_train

    
    def inference(self, x):
        """
        forward network
        """
        layers = []
        feats = []
        
        D = x.get_shape()[3]
        output_ch = 32
        num_classes = coco_input.NUM_CLASSES
        feat = x

        # 28x28
        for idx_layer in range(5):
            name = "layer{}".format(idx_layer)

            if idx_layer == 0:
                layer = Layer(name, output_ch, retain_ratio=0.8, is_train=self.is_train)
            else:
                layer = Layer(name, output_ch, is_train=self.is_train)

            feat = layer.inference(feat)
            output_ch *= 2

            feats.append(feat)
            layers.append(layer)


        self.conv_outputs = feats
        self.conv_layers = layers


        # Global Average Pooling
        with tf.variable_scope("GAP"):
            N, H, W, C = feat.get_shape()
            w = _var("W", [H,W,C,num_classes])
            b = _var("b", [num_classes],initializer=tf.constant_initializer())
                    
            feat = tf.nn.conv2d(feat, w, strides=[1,1,1,1],padding="VALID")
            logits = feat + b
            logits = tf.contrib.layers.flatten(logits)
    
        return logits


def get_loss(labels, logits):
    vector_labels = tf.one_hot(labels, coco_input.NUM_CLASSES, dtype=tf.float32)
    
    entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits, vector_labels), name="entropy")
    decays = tf.add_n(tf.get_collection('losses'), name="weight_loss")
    total_loss = tf.add(entropy, decays, "total_loss")

    tf.scalar_summary("entropy", entropy)
    tf.scalar_summary("total_loss",total_loss)
    
    return entropy, total_loss
