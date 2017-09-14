# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 11:53:38 2017

@author: Weiyu_Lee
"""
import time
import os
import sys
sys.path.append('./utility')

from utils import (
  read_data, 
  input_setup, 
  imsave,
  merge,
  imread,
  psnr
)

import tensorflow as tf
import random

import model_zoo

class SRCNN(object):

    def __init__(self, 
               sess, 
               image_size=33,
               label_size=21, 
               batch_size=128,
               color_dim=1, 
               checkpoint_dir=None, 
               output_dir=None,
               is_train=True):
        """
        Initial function
          
        Args:
            image_size: training or testing input image size. 
                        (if scale=3, image size is [33x33].)
            label_size: label image size. 
                        (if scale=3, image size is [21x21].)
            batch_size: batch size
            color_dim: color dimension number. (only Y channel, color_dim=1)
            checkpoint_dir: checkpoint directory
            output_dir: output directory
        """
        self.sess = sess
        self.is_grayscale = (color_dim == 1)
        self.image_size = image_size
        self.label_size = label_size
        self.batch_size = batch_size

        self.color_dim = color_dim
    
        self.checkpoint_dir = checkpoint_dir
        self.output_dir = output_dir
        
        self.is_train = is_train
        
        self.build_model()

    def build_model(self):###
        """
        Build SRCNN model
        """        
        # Define input and label images
        self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.color_dim], name='images')
        self.labels = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.color_dim], name='labels')
        self.dropout = tf.placeholder(tf.float32, name='dropout')
       
        mz = model_zoo.model_zoo(self.images, self.dropout, self.is_train, "srcnn_v1")
        
        # Build model
        self.pred = mz.build_model()
    
        # Define loss function (MSE) 
        self.loss = tf.reduce_mean(tf.square(self.labels - self.pred))
    
        self.saver = tf.train.Saver()

    def train(self, config):
        """
        Training process.
        """     
        print("Training...")
    
        input_setup(self.sess, config)
       
        data_dir = os.path.join('./{}'.format(config.checkpoint_dir), "train.h5")
    
        train_data, train_label = read_data(data_dir)
    
        # Stochastic gradient descent with the standard backpropagation
        #self.train_op = tf.train.GradientDescentOptimizer(config.learning_rate).minimize(self.loss)
        self.train_op = tf.train.AdamOptimizer(config.learning_rate).minimize(self.loss)
    
        self.sess.run(tf.global_variables_initializer())

        # Define iteration counter, timer and average loss
        itera_counter = 0
        avg_loss = 0
        avg_500_loss = 0
        start_time = time.time()   
        
        # Load checkpoint 
        if self.load(self.checkpoint_dir, config.scale):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
             
        for ep in range(config.epoch):
            # Run by batch images
            batch_idxs = len(train_data) // config.batch_size
              
            # Shuffle the batch data
            shuffled_data = list(zip(train_data, train_label))
            random.shuffle(shuffled_data)
            train_data, train_label = zip(*shuffled_data)
            
            for idx in range(0, batch_idxs):
                itera_counter += 1
                  
                # Get the training and testing data
                batch_images = train_data[idx*config.batch_size : (idx+1)*config.batch_size]
                batch_labels = train_label[idx*config.batch_size : (idx+1)*config.batch_size]
                  
                # Run the model
                _, err = self.sess.run([self.train_op, self.loss], 
                                       feed_dict={
                                                   self.images: batch_images, 
                                                   self.labels: batch_labels,
                                                   self.dropout: 1.
                                                 })
    
                avg_loss += err
                avg_500_loss += err
    
                if itera_counter % 10 == 0:
                    print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" \
                         % ((ep+1), itera_counter, time.time()-start_time, err))
    
                if itera_counter % 500 == 0:
                    self.save(config.checkpoint_dir, config.scale, itera_counter)
                
                    print("==> Epoch: [%2d], average loss of 500 steps: [%.8f], average loss: [%.8f]" \
                         % ((ep+1), avg_500_loss/500, avg_loss/itera_counter))            
                    avg_500_loss = 0
    
    def test(self, config):
        """
        Testing process.
        """          
        print("Testing...")

        # Load checkpoint        
        if self.load(self.checkpoint_dir, config.scale):
          print(" [*] Load SUCCESS")
        else:
          print(" [!] Load failed...")        
        
        nx, ny = input_setup(self.sess, config)
        
        data_dir = os.path.join('./{}'.format(config.checkpoint_dir), "test.h5")
        
        test_data, test_label = read_data(data_dir)
           
        result = self.pred.eval({
                                    self.images: test_data, 
                                    self.labels: test_label,
                                    self.dropout: 1.
                                 })
        
        result = merge(result, [nx, ny])
        result = result.squeeze()
        
        # Save output image
        output_path = os.path.join(os.getcwd(), config.output_dir)
        image_path = os.path.join(output_path, "test_img.png")
        imsave(result, image_path)
        
        # PSNR
        label_path = os.path.join(output_path, "test_org_img.png")
        bicubic_path = os.path.join(output_path, "test_bicubic_img.png")
        
        bicubic_img = imread(bicubic_path, is_grayscale=True)
        label_img = imread(label_path, is_grayscale=True)
        output_img = imread(image_path, is_grayscale=True)
        
        bicubic_psnr_value = psnr(label_img, bicubic_img)        
        srcnn_psnr_value = psnr(label_img, output_img)        
        
        print("Bicubic PSNR: [{}]".format(bicubic_psnr_value))
        print("SRCNN PSNR: [{}]".format(srcnn_psnr_value))
        
#    def model(self):
#        """
#        Testing process.
#        To avoid border effects during training, all the convolutional layers 
#        have no padding, and the network produces a smaller output.
#        """          
#        
#        # Layer 1: Patch extraction and representation
#        conv1 = tf.nn.relu(tf.nn.conv2d(self.images, self.weights['w1'], strides=[1,1,1,1], padding='VALID') + self.biases['b1'])
#        
#        # Layer 2: Non-linear mapping
#        conv2 = tf.nn.relu(tf.nn.conv2d(conv1, self.weights['w2'], strides=[1,1,1,1], padding='VALID') + self.biases['b2'])
#        
#        # Layer 3: Reconstruction
#        conv3 = tf.nn.conv2d(conv2, self.weights['w3'], strides=[1,1,1,1], padding='VALID') + self.biases['b3']
#        
#        return conv3

    def save(self, checkpoint_dir, scale, step):
        """
        Save the checkpoint. 
        According to the scale, use different folder to save the models.
        """          
        
        print(" [*] Saving checkpoints...step: [{}]".format(step))
        model_name = "SRCNN.model"
        model_dir = "%s_%s_%s" % ("srcnn", "scale", scale)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
    
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
    
        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir, scale):
        """
        Load the checkpoint. 
        According to the scale, read different folder to load the models.
        """     
        
        print(" [*] Reading checkpoints...")
        model_dir = "%s_%s_%s" % ("srcnn", "scale", scale)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
    
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
        
            return True
        
        else:
            return False
