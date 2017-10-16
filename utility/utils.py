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

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

def read_data(path):  
    """
    Read h5 format data file
      
    Args:
        path: file path of desired file
        data: '.h5' file format that contains train data values
        label: '.h5' file format that contains train label values
    """
    
    with h5py.File(path, 'r') as hf:
        data = np.array(hf.get('data'))
        label = np.array(hf.get('label'))
        
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
        input_: image applied bicubic interpolation (low-resolution)
        label_: image with original resolution (high-resolution)
    """

    # Read Image
    input_ = imread(input_path, is_grayscale=True)
    label_ = imread(label_path, is_grayscale=True)
   
    input_ = modcrop(input_, scale)
    label_ = modcrop(label_, scale)
    
    # Normalization
    input_ = input_ / 255.
    label_ = label_ / 255.
    
    return input_, label_

def get_file_path(sess, dataset_folder, child_folder=""):
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
    label_data_ext = "*_org.bmp"
    input_data_ext = "*_bicubic_scale_{}.bmp".format(FLAGS.scale)
    
    if FLAGS.is_train:
        data_dir = os.path.join(os.getcwd(), dataset_folder, child_folder, preprocessed_folder)
        label_data = glob.glob(os.path.join(data_dir, label_data_ext))
        input_data = glob.glob(os.path.join(data_dir, input_data_ext))
    else:
        data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset_folder)), child_folder, preprocessed_folder)
        label_data = glob.glob(os.path.join(data_dir, label_data_ext))
        input_data = glob.glob(os.path.join(data_dir, input_data_ext))
    
    label_data = sorted(label_data)
    input_data = sorted(input_data)
       
    return input_data, label_data

def make_data(sess, data, label):
    """
    Make input data as h5 file format
    Depending on 'is_train' (flag value), savepath would be changed.
    """
    
    if FLAGS.is_train:
        savepath = os.path.join(os.getcwd(), 'checkpoint/train.h5')
    else:
        savepath = os.path.join(os.getcwd(), 'checkpoint/test.h5')
    
    with h5py.File(savepath, 'w') as hf:
        hf.create_dataset('data', data=data)
        hf.create_dataset('label', data=label)

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

def input_setup(sess, config):
    """
    Read image files and make their sub-images and saved them as a h5 file format.
    """
  
    # Load data path
    if config.is_train:
        input_data, label_data = get_file_path(sess, dataset_folder="Train")
    else:
        input_data, label_data = get_file_path(sess, dataset_folder="Test", child_folder=config.test_folder)

    sub_input_sequence = []
    sub_label_sequence = []
    
    # Calculate the padding size
    padding = int(abs(config.image_size - config.label_size) / 2) # 6  

    if config.is_train:            
        for i in range(len(input_data)):
            # Preprocess the input images
            input_, label_ = preprocess(input_data[i], label_data[i], config.scale)
            if len(input_.shape) == 3:
                h, w, _ = input_.shape
            else:
                h, w = input_.shape
            
            # Crop the input images
            for x in range(0, h-config.image_size+1, config.extract_stride):
                for y in range(0, w-config.image_size+1, config.extract_stride):
                    sub_input = input_[x:x+config.image_size, y:y+config.image_size] # [33 x 33]
                    sub_label = label_[x+padding:x+padding+config.label_size, y+padding:y+padding+config.label_size] # [21 x 21]
    
                    # Make channel value
                    sub_input = sub_input.reshape([config.image_size, config.image_size, 1])  
                    sub_label = sub_label.reshape([config.label_size, config.label_size, 1])
    
                    sub_input_sequence.append(sub_input)
                    sub_label_sequence.append(sub_label)

    else:       
        # Preprocess the input images
        input_, label_ = preprocess(input_data[2], label_data[2], config.scale)
    
        if len(input_.shape) == 3:
            h, w, _ = input_.shape
        else:
            h, w = input_.shape
    
        # Crop the input images
        # Numbers of sub-images in height and width of image are needed to compute merge operation.
        nx = ny = 0       
        for x in range(0, h-config.image_size+1, config.extract_stride):
            nx += 1; ny = 0
            for y in range(0, w-config.image_size+1, config.extract_stride):
                ny += 1
                sub_input = input_[x:x+config.image_size, y:y+config.image_size] # [33 x 33]
                sub_label = label_[x+padding:x+padding+config.label_size, y+padding:y+padding+config.label_size] # [21 x 21]
                
                sub_input = sub_input.reshape([config.image_size, config.image_size, 1])  
                sub_label = sub_label.reshape([config.label_size, config.label_size, 1])
        
                sub_input_sequence.append(sub_input)
                sub_label_sequence.append(sub_label)

        # Save the input images
        # According to the size of label images, crop the suitable size 
        # that match with the output. 
        print("Saving the original images... size: [{} x {}]".format(config.extract_stride*nx, config.extract_stride*ny))
        image_path = os.path.join(os.getcwd(), config.output_dir)
        org_img_path = os.path.join(image_path, "test_org_img.png")      
        bicubic_img_path = os.path.join(image_path, "test_bicubic_img.png")                 
        imsave(label_[padding:padding+config.extract_stride*nx, padding:padding+config.extract_stride*ny], org_img_path)
        imsave(input_[padding:padding+config.extract_stride*nx, padding:padding+config.extract_stride*ny], bicubic_img_path)

    """
    len(sub_input_sequence) : the number of sub_input (33 x 33 x ch) in one image
    (sub_input_sequence[0]).shape : (33, 33, 1)
    """
    # Make list to numpy array. With this transform
    arrdata = np.asarray(sub_input_sequence) # [?, 33, 33, 1]
    arrlabel = np.asarray(sub_label_sequence) # [?, 21, 21, 1]

    make_data(sess, arrdata, arrlabel)

    if not config.is_train:
        return nx, ny

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

def imsave(image, path, max_I=255, is_normalized=False):   
    """
    Save the image by scipy.misc.toimage()
    
    Note: 
        Default maximum of the data is 255. (8-bit)
        Should not use scipy.misc.imsave() to save image, 
        it might casue info. loss due to the data type.        
    """
    if is_normalized==True: image = image*max_I
    print(np.max(image))
    output_image = scipy.misc.toimage(image, high=np.max(image), low=np.min(image), mode='L')
    output_image.save(path)
    #return scipy.misc.imsave(path, image)
    
def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h*size[0], w*size[1], 1))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image
    
    return img
