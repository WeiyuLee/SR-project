# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 11:51:21 2017

@author: Weiyu_Lee
"""

from srcnn_model import SRCNN

import tensorflow as tf

import pprint
import os

flags = tf.app.flags
flags.DEFINE_integer("epoch", 10, "Number of epoch [10]")
flags.DEFINE_integer("batch_size", 128, "The size of batch images [128]")
flags.DEFINE_integer("image_size", 32, "The size of image to use [33]")
flags.DEFINE_integer("label_size", 20, "The size of label to produce [21]")
flags.DEFINE_float("learning_rate", 1e-4, "The learning rate of gradient descent algorithm [1e-4]")
flags.DEFINE_integer("color_dim", 1, "Dimension of image color. [1]")
flags.DEFINE_integer("scale", 4, "The size of scale factor for preprocessing input image [3]")
flags.DEFINE_integer("extract_stride", 14, "The size of stride to apply input image [14]")

flags.DEFINE_string("checkpoint_dir", "checkpoint", "Name of checkpoint directory [checkpoint]")
flags.DEFINE_string("output_dir", "output", "Name of sample directory [output]")
flags.DEFINE_string("train_dir", "Train", "Name of train dataset directory")
flags.DEFINE_string("test_dir", "Test/Set5", "Name of test dataset directory [Test/Set5]")

flags.DEFINE_string("train_h5_name", "train.h5", "Name of train dataset .h5 file")
flags.DEFINE_string("test_h5_name", "test.h5", "Name of test dataset .h5 file")

flags.DEFINE_boolean("is_train", True, "True for training, False for testing [True]")
FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()

def main(_):
  pp.pprint(flags.FLAGS.__flags)

  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.output_dir):
    os.makedirs(FLAGS.output_dir)

  with tf.Session() as sess:
    srcnn = SRCNN(sess, 
                  image_size=FLAGS.image_size, 
                  label_size=FLAGS.label_size, 
                  batch_size=FLAGS.batch_size,
                  color_dim=FLAGS.color_dim, 
                  checkpoint_dir=FLAGS.checkpoint_dir,
                  output_dir=FLAGS.output_dir,
                  is_train=FLAGS.is_train)

    if FLAGS.is_train:
        srcnn.train(FLAGS)
    else:
        srcnn.test(FLAGS)
    
if __name__ == '__main__':
  tf.app.run()