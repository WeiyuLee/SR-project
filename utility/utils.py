# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 16:03:19 2017

@author: Weiyu_Lee
"""

"""
Scipy version > 0.18 is needed, due to 'mode' option from scipy.misc.imread function
"""

import os
import glob
import h5py

import scipy.misc
import scipy.ndimage
import numpy as np
import random

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

def read_data(path, stage_size):  
    """
    Read h5 format data file
      
    Args:
        path: file path of desired file
        data: '.h5' file format that contains train data values
        label: '.h5' file format that contains train label values
    """
    
    with h5py.File(path, 'r') as hf:
        data = np.array(hf.get('data'))
        
        label = [None]*stage_size
        for i in range(stage_size):
            label[i] = np.array(hf.get('stg_{}_label'.format(i)))
        
    return data, label

def normalization(image, mean, stddev):
    """
    Normalize the input image.
    Return ndarray
    """    

    # Source code's normalization method
    image = image / 255.
    
    return image

def preprocess(input_path, label_path, scale=3):
    """
    Preprocess single image file 
        (1) Read original image as YCbCr format (and grayscale as default)
        (2) Normalize
        (3) Apply image file with bicubic interpolation
    
    Args:
        path: file path of desired file
        input: image applied bicubic interpolation (low-resolution)
        label: image with original resolution (high-resolution)
    """

    # Read Image
    input = imread(input_path, is_grayscale=True)
    
    label = []
    for i in range(len(label_path)):
        label.append(imread(label_path[i], is_grayscale=True))
        label[i] = modcrop(label[i], scale)
        label[i] = label[i] / 255.
   
    input = modcrop(input, scale)
    #label = modcrop(label, scale)
    
    # Normalization
    input = input / 255.
    #label = label / 255.
    
    return input, label

def get_file_path(sess, stage_size, dataset_folder):
    """
    Get the dataset file path.
    According to the scale, choose the correct folder that made by Matlab preprocessing code.
    
    Args:
        dataset: choose train dataset or test dataset
        
        For train dataset, output data would be ['.../t1.bmp', '.../t2.bmp', ..., '.../t99.bmp']
    """
    
    # Define the folder name
    preprocessed_folder = "preprocessed_scale_{}".format(FLAGS.scale)
    
    # Define the preprocessed ext.
    label_data_ext = []
    for i in range(1, stage_size+1):
        label_data_ext.append("*_stg_{}.bmp".format(i))
           
    input_data_ext = "*_bicubic_scale_{}_input.bmp".format(FLAGS.scale)

    data_dir = os.path.join(os.getcwd(), dataset_folder, preprocessed_folder)

    label_data = [None]*stage_size
    for i in range(stage_size):
        label_data[i] = glob.glob(os.path.join(data_dir, label_data_ext[i]))
        label_data[i] = sorted(label_data[i])
           
    input_data = glob.glob(os.path.join(data_dir, input_data_ext))

    org_data = glob.glob(os.path.join(dataset_folder, "*.bmp"))
    
    input_data = sorted(input_data)
    org_data = sorted(org_data)   
   
    return input_data, label_data, org_data

def make_data(data, label, save_dir):
    """
    Make input data as h5 file format
    """
    
    savepath = os.path.join(os.getcwd(), save_dir)
    
    with h5py.File(savepath, 'w') as hf:
        hf.create_dataset('data', data=data)
        for i in range(len(label)):
            hf.create_dataset('stg_{}_label'.format(i), data=label[i])

def imread(path, is_grayscale=True):
    """
    Read image using its path.
    Default value is gray-scale, and image is read by YCbCr format as the paper said.
    Return ndarray
    """
    
    if is_grayscale:
        return scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float64)
    else:
        return scipy.misc.imread(path, mode='YCbCr').astype(np.float64)

def modcrop(image, scale=3):
    """
    In order to scale down and up the original image, the first thing needed to
    do is to have no remainder while scaling operation.
    We need to find modulo of height (and width) and scale factor.
    Then, subtract the modulo from height (and width) of original image size.
    There would be no remainder even after scaling operation.
    """
  
    if len(image.shape) == 3:
        h, w, _ = image.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        image = image[0:h, 0:w, :]
    else:
        h, w = image.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        image = image[0:h, 0:w]
        
    return image

def input_setup(sess, data_dir, save_dir, extract_stride, is_save, config):
    """
    Read image files and make their sub-images and saved them as a h5 file format.
    """

    input_data, label_data, org_data = get_file_path(sess, config.stage_size, dataset_folder=data_dir)        

    sub_input_sequence = []
    sub_label_sequence = [[] for i in range(config.stage_size)]
    
    # Calculate the padding size
    padding = int(abs(config.image_size - config.label_size) / 2) # 6                 

    ## nxs, nys: Record the patch number of every test image 
    nxs = []
    nys = []

    for i in range(len(input_data)):
        # Preprocess the input images
        input_, label_ = preprocess(input_data[i], [row[i] for row in label_data], config.scale)
    
        if len(input_.shape) == 3:
            h, w, _ = input_.shape
        else:
            h, w = input_.shape
    
        # Crop the input images
        # Numbers of sub-images in height and width of image are needed to compute merge operation.
        sub_label = [None]*config.stage_size
        nx = ny = 0       
        for x in range(0, h-config.image_size+1, extract_stride):
            nx += 1; ny = 0
            for y in range(0, w-config.image_size+1, extract_stride):
                ny += 1
                
                sub_input = input_[x:x+config.image_size, y:y+config.image_size] # [33 x 33]
                for n_stg in range(config.stage_size):
                    if not n_stg == (config.stage_size - 1):
                        sub_label[n_stg] = label_[-1][x:x+config.image_size, y:y+config.image_size]###########################
                        sub_label[n_stg] = sub_label[n_stg].reshape([config.image_size, config.image_size, 1])
                    else:
                        sub_label[n_stg] = label_[-1][x+padding:x+padding+config.label_size, y+padding:y+padding+config.label_size]###########################
                        sub_label[n_stg] = sub_label[n_stg].reshape([config.label_size, config.label_size, 1])
                    
                    sub_label_sequence[n_stg].append(sub_label[n_stg])

                # Make channel value
                sub_input = sub_input.reshape([config.image_size, config.image_size, 1])  
                sub_input_sequence.append(sub_input)
                
        if is_save:
            # Save the input images
            # According to the size of label images, crop the suitable size 
            # that match with the output. 
            print("Saving the original images... size: [{} x {}] -> [{} x {}]".format(h, w, extract_stride*nx, extract_stride*ny))
            
            print("nx = {}, ny = {}".format(nx, ny))
            
            # Find the file name and ext
            base = os.path.basename(org_data[i])
            output_filename, output_ext = os.path.splitext(base)
            output_path = os.path.join(os.getcwd(), config.output_dir)
            
            # Combine the path str
            org_img_path = os.path.join(output_path, output_filename + "_org_img" + output_ext)      
            bicubic_img_path = os.path.join(output_path, output_filename + "_bicubic_img" + output_ext)                 
            
            # Save images
            imsave(label_[-1][padding:padding+extract_stride*nx, padding:padding+extract_stride*ny], org_img_path)
            imsave(input_[padding:padding+extract_stride*nx, padding:padding+extract_stride*ny], bicubic_img_path)
        
        # Record the patch number
        nxs.append(nx)
        nys.append(ny)

    # Make list to numpy array. With this transform
    arrlabel = [None]*config.stage_size
    arrdata = np.asarray(sub_input_sequence) # [?, 33, 33, 1]
    for i in range(config.stage_size):
        arrlabel[i] = (np.asarray(sub_label_sequence[i])) # [?, 21, 21, 1]

    make_data(arrdata, arrlabel, save_dir)

    return nxs, nys, org_data

def batch_shuffle(data, label, batch_size):
    """
    Shuffle the batch data
    """
    # Shuffle the batch data
    shuffled_data = list(zip(data, *label))
    random.shuffle(shuffled_data)
    tmp = list(zip(*shuffled_data))
    
    data_shuffled = tmp[0]
    label_shuffled = tmp[1:]
    
    return data_shuffled, label_shuffled

def psnr(img1, img2, max_I=255):
    """
    Calculate PSNR value.
    
    Note: 
        Default maximum of the data is 255. (8-bit)
    """
    mse = np.square(img1 - img2)
    mse = mse.mean()
    psnr = 20*np.log10(max_I/(np.sqrt(mse)))    
    
    return psnr

def imsave(image, path, max_I=255):   
    """
    Save the image by scipy.misc.toimage()
    
    Note: 
        Default maximum of the data is 255. (8-bit)
        Should not use scipy.misc.imsave() to save image, 
        it might casue info. loss due to the data type.        
    """
    image = image*max_I
    output_image = scipy.misc.toimage(image, high=np.max(image), low=np.min(image), mode='L')
    output_image.save(path)
    
def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h*size[0], w*size[1], 1))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image
    
    return img
