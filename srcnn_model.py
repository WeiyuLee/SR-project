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
  psnr,
  batch_shuffle
)

import tensorflow as tf

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

        # Define dataset path
        train_data_dir = os.path.join('./{}'.format(config.checkpoint_dir), config.train_h5_name)
        test_data_dir = os.path.join('./{}'.format(config.checkpoint_dir), config.test_h5_name)

        # If dataset not exists, create it.
        if not os.path.exists(train_data_dir):
            print("Preparing training dataset...")
            save_dir = os.path.join(os.getcwd(), config.checkpoint_dir, config.train_h5_name)
            input_setup(self.sess, config.train_dir, save_dir, config)
        if not os.path.exists(test_data_dir):
            print("Preparing testing dataset...")
            save_dir = os.path.join(os.getcwd(), config.checkpoint_dir, config.test_h5_name)
            input_setup(self.sess, config.test_dir, save_dir, config)
        
        # Read data from .h5 file
        train_data, train_label = read_data(train_data_dir)
        test_data, test_label = read_data(test_data_dir)
    
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
            train_batch_idxs = len(train_data) // config.batch_size
            train_data, train_label = batch_shuffle(train_data, train_label, config.batch_size)
            
            for idx in range(0, train_batch_idxs):
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
                    
                    # Validation
                    ## Run the test images
                    _, test_err = self.sess.run([self.train_op, self.loss], 
                                           feed_dict={
                                                       self.images: test_data, 
                                                       self.labels: test_label,
                                                       self.dropout: 1.
                                                     })
                    print("==> Epoch: [%2d], average loss of 500 steps: [%.8f], average loss: [%.8f], validation loss: [%.8f]" \
                         % ((ep+1), avg_500_loss/500, avg_loss/itera_counter, test_err))            
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
        
        # Preparing testing dataset (save the cropped image and .h5 file)        
        save_dir = os.path.join(os.getcwd(), config.checkpoint_dir, config.test_h5_name)
        nxs, nys, org_data = input_setup(self.sess, config.test_dir, save_dir, config)

        # Read data from .h5 file
        data_dir = os.path.join('./{}'.format(config.checkpoint_dir), "test.h5")
        test_data, test_label = read_data(data_dir)
           
        result = self.pred.eval({
                                    self.images: test_data, 
                                    self.labels: test_label,
                                    self.dropout: 1.
                                 })
        
        # Run all the test images
        idx = 0 # record the patches' indeies 
        for i in range(len(nxs)):
            tmp_img = merge(result[idx:idx+nxs[i]*nys[i], :, :, :], [nxs[i], nys[i]])
            tmp_img = tmp_img.squeeze()
            
            # Save output image
            base = os.path.basename(org_data[i])
            output_filename, output_ext = os.path.splitext(base)
            output_dir = os.path.join(os.getcwd(), config.output_dir)
                      
            test_path = os.path.join(output_dir, output_filename + "_test_img" + output_ext)
            imsave(tmp_img, test_path)
            
            # PSNR
            ## Read from the output dir. to calculated PSNR value
            label_path = os.path.join(output_dir, output_filename + "_org_img" + output_ext)
            bicubic_path = os.path.join(output_dir, output_filename + "_bicubic_img" + output_ext)
            
            bicubic_img = imread(bicubic_path, is_grayscale=True)
            label_img = imread(label_path, is_grayscale=True)
            test_img = imread(test_path, is_grayscale=True)
            
            bicubic_psnr_value = psnr(label_img, bicubic_img)        
            srcnn_psnr_value = psnr(label_img, test_img)        
            
            print("[{}] Bicubic PSNR: [{}]".format(output_filename, bicubic_psnr_value))
            print("[{}] SRCNN PSNR: [{}]".format(output_filename, srcnn_psnr_value))
        
            ## Update index
            idx += nxs[i]*nys[i]

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
