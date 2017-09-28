# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 11:53:38 2017

@author: Weiyu_Lee
"""
import os
import sys
sys.path.append('./utility')

from tqdm import tqdm

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
               image_size=32,
               label_size=20, 
               batch_size=128,
               color_dim=1, 
               scale=4,
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
        
        self.scale = scale
    
        self.checkpoint_dir = checkpoint_dir
        self.output_dir = output_dir
        
        self.is_train = is_train
        
        # Used for recording testing patches' assemble information
        self.nxs = []
        self.nys = []
        self.org_data = []
        
        self.build_model()

    def build_model(self):###
        """
        Build SRCNN model
        """        
        # Define input and label images
        self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.color_dim], name='images')
        self.stg1_labels = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.color_dim], name='stg1_labels')
        self.stg2_labels = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.color_dim], name='stg2_labels')
        self.stg3_labels = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.color_dim], name='stg3_labels')
        self.dropout = tf.placeholder(tf.float32, name='dropout')
       
        mz = model_zoo.model_zoo(self.images, self.dropout, self.is_train, "grr_srcnn_v1")
        
        # Build model
        self.pred = mz.build_model()
           
        # Define loss function (MSE) 
        ## Stage 1 loss:
        self.stg1_loss = tf.reduce_mean(tf.square(self.stg1_labels - self.pred[0]))
        ## Stage 2 loss:
        self.stg2_loss = tf.reduce_mean(tf.square(self.stg2_labels - self.pred[1]))    
        ## Stage 3 loss:
        self.stg3_loss = tf.reduce_mean(tf.square(self.stg3_labels - self.pred[2]))
    
        self.loss = tf.add(tf.add(self.stg1_loss, self.stg2_loss), self.stg3_loss)
    
        self.saver = tf.train.Saver()

    def prepare_data(self, config):
        
        print("Preparing training dataset...")
        save_dir = os.path.join(os.getcwd(), config.checkpoint_dir, config.train_h5_name)
        input_setup(self.sess, config.train_dir, save_dir, config.train_extract_stride, False, config)

        print("Preparing testing dataset...")
        save_dir = os.path.join(os.getcwd(), config.checkpoint_dir, config.test_h5_name)
        self.nxs, self.nys, self.org_data = input_setup(self.sess, config.test_dir, save_dir, config.test_extract_stride, True, config)

    def train(self, config):
        """
        Training process.
        """     
        print("Training...")

        # Define dataset path
        train_data_dir = os.path.join('./{}'.format(config.checkpoint_dir), config.train_h5_name)
        test_data_dir = os.path.join('./{}'.format(config.checkpoint_dir), config.test_h5_name)
        
        # Read data from .h5 file
        train_data, train_label = read_data(train_data_dir, config.stage_size)
        test_data, test_label = read_data(test_data_dir, config.stage_size)
    
        # Stochastic gradient descent with the standard backpropagation       
        self.train_op = tf.train.AdamOptimizer(config.learning_rate).minimize(self.loss)
    
        self.sess.run(tf.global_variables_initializer())

        # Define iteration counter, timer and average loss
        itera_counter = 0
        avg_500_loss = [0]*(config.stage_size+1)
        
        # Load checkpoint 
        if self.load_ckpt(self.checkpoint_dir, config.ckpt_name):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
             
        batch_labels = [None]*config.stage_size            
                   
        train_batch_num = len(train_data) // config.batch_size
        
        epoch_pbar = tqdm(range(config.epoch))
        for ep in epoch_pbar:            
            # Run by batch images
            train_data, train_label = batch_shuffle(train_data, train_label, config.batch_size)
        
            epoch_pbar.set_description("Epoch: [%2d]" % ((ep+1)))
            epoch_pbar.refresh()
        
            batch_pbar = tqdm(range(0, train_batch_num), desc="Batch: [0]")
            for idx in batch_pbar:                
                itera_counter += 1
                  
                # Get the training and testing data
                batch_images = train_data[idx*config.batch_size : (idx+1)*config.batch_size]
                
                batch_labels[0] = train_label[0][idx*config.batch_size : (idx+1)*config.batch_size]
                batch_labels[1] = train_label[1][idx*config.batch_size : (idx+1)*config.batch_size]
                batch_labels[2] = train_label[2][idx*config.batch_size : (idx+1)*config.batch_size]
                                 
                # Run the model
                _, stg1_err, stg2_err, stg3_err, total_err = self.sess.run([  self.train_op, 
                                                                              self.stg1_loss, 
                                                                              self.stg2_loss, 
                                                                              self.stg3_loss,
                                                                              self.loss], 
                                                                              feed_dict={
                                                                                          self.images: batch_images, 
                                                                                          self.stg1_labels: batch_labels[0],
                                                                                          self.stg2_labels: batch_labels[1],
                                                                                          self.stg3_labels: batch_labels[2],
                                                                                          self.dropout: 1.
                                                                                        })   

                avg_500_loss[0] += stg1_err
                avg_500_loss[1] += stg2_err
                avg_500_loss[2] += stg3_err
                avg_500_loss[3] += total_err
    
                batch_pbar.set_description("Batch: [%2d]" % (idx+1))
                batch_pbar.refresh()
            
            if ep % 5 == 0:
                self.save_ckpt(config.checkpoint_dir, config.ckpt_name, itera_counter)
                
                # Validation
                ## Run the test images
                val_stg1_err, val_stg2_err, val_stg3_err, val_total_err = self.sess.run([self.stg1_loss,
                                                                                     self.stg2_loss,
                                                                                     self.stg3_loss,
                                                                                     self.loss], 
                                                                                     feed_dict={
                                                                                                 self.images: test_data, 
                                                                                                 self.stg1_labels: test_label[0],
                                                                                                 self.stg2_labels: test_label[1],
                                                                                                 self.stg3_labels: test_label[2],
                                                                                                 self.dropout: 1.
                                                                                               })
 
                for i in range(len(avg_500_loss)):
                    avg_500_loss[i] /= (train_batch_num*5)
                print("Epoch: [%2d], Average loss of 500 steps: stg loss: [%.8f, %.8f, %.8f], total loss: [%.8f]" \
                     % ((ep+1), avg_500_loss[0], avg_500_loss[1], avg_500_loss[2], avg_500_loss[3]))               
                print("Epoch: [%2d], Test stg loss: [%.8f, %.8f, %.8f], total loss: [%.8f]\n" \
                      % ((ep+1), val_stg1_err, val_stg2_err, val_stg3_err, val_total_err))       
                
                avg_500_loss = [0]*(config.stage_size+1)                                     
            
    def test(self, config):
        """
        Testing process.
        """          
        print("Testing...")

        # Load checkpoint        
        if self.load_ckpt(self.checkpoint_dir, config.ckpt_name):
          print(" [*] Load SUCCESS")
        else:
          print(" [!] Load failed...")        
        
        # Read data from .h5 file
        test_data_dir = os.path.join('./{}'.format(config.checkpoint_dir), config.test_h5_name)
        test_data, test_label = read_data(test_data_dir, config.stage_size)
           
        result = self.pred[2].eval({
                                   self.images: test_data, 
                                   self.stg1_labels: test_label[0],
                                   self.stg2_labels: test_label[1],
                                   self.stg3_labels: test_label[2],
                                   self.dropout: 1.
                                 })
        
        print(self.nxs)
    
        # Run all the test images
        idx = 0 # record the patches' indeies 
        avg_psnr = 0
        for i in range(len(self.nxs)):
            tmp_img = merge(result[idx:idx+self.nxs[i]*self.nys[i], :, :, :], [self.nxs[i], self.nys[i]])
            tmp_img = tmp_img.squeeze()
            
            print(self.org_data[i])
            print("nxs[{}] = {}, nys[{}] = {}".format(i, self.nxs[i], i, self.nys[i]))
            
            # Save output image
            base = os.path.basename(self.org_data[i])
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
            idx += self.nxs[i]*self.nys[i]
            
            ## Update avg. psnr value
            avg_psnr += srcnn_psnr_value

        print("Average PSNR: [{}]".format(avg_psnr/len(self.nxs)))

    def save_ckpt(self, checkpoint_dir, ckpt_name, step):
        """
        Save the checkpoint. 
        According to the scale, use different folder to save the models.
        """          
        
        print(" [*] Saving checkpoints...step: [{}]".format(step))
        model_name = "SRCNN.model"
        
        if ckpt_name == "":
            model_dir = "%s_%s_%s" % ("srcnn", "scale", self.scale)
        else:
            model_dir = ckpt_name
        
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
    
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
    
        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load_ckpt(self, checkpoint_dir, ckpt_name=""):
        """
        Load the checkpoint. 
        According to the scale, read different folder to load the models.
        """     
        
        print(" [*] Reading checkpoints...")
        if ckpt_name == "":
            model_dir = "%s_%s_%s" % ("srcnn", "scale", self.scale)
        else:
            model_dir = ckpt_name
            
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
    
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
        
            return True
        
        else:
            return False
