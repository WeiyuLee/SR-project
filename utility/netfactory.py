import tensorflow as tf
import numpy as np

import sys
sys.path.append('./utility')
#import cifar10
import utility as ut


def convolution_layer(inputs, kernel_shape, stride, name, flatten = False ,padding = 'SAME',initializer=tf.contrib.layers.xavier_initializer(), activat_fn=tf.nn.relu):
    
    pre_shape = inputs.get_shape()[-1]
    rkernel_shape = [kernel_shape[0], kernel_shape[1], pre_shape, kernel_shape[2]]     
    
    with tf.variable_scope(name) as scope:
        
        try:
            weight = tf.get_variable("weights",rkernel_shape, tf.float32, initializer=initializer)
            bias = tf.get_variable("bias",kernel_shape[2], tf.float32, initializer=initializer)
        except:
            scope.reuse_variables()
            weight = tf.get_variable("weights",rkernel_shape, tf.float32, initializer=initializer)
            bias = tf.get_variable("bias",kernel_shape[2], tf.float32, initializer=initializer)
        
        net = tf.nn.conv2d(inputs, weight,stride, padding=padding)
        net = tf.add(net, bias)
        
        if not activat_fn==None:
            net = activat_fn(net, name=name+"_out")
        
        if flatten == True:
            net = tf.reshape(net, [-1, int(np.prod(net.get_shape()[1:]))], name=name+"_flatout")
        
    return net


def fc_layer(inputs, out_shape, name,initializer=tf.contrib.layers.xavier_initializer(), activat_fn=tf.nn.relu):
    
    
    pre_shape = inputs.get_shape()[-1]
    
    with tf.variable_scope(name) as scope:
        
        
        try:
            weight = tf.get_variable("weights",[pre_shape, out_shape], tf.float32, initializer=initializer)
            bias = tf.get_variable("bias",out_shape, tf.float32, initializer=initializer)
        except:
            scope.reuse_variables()
            weight = tf.get_variable("weights",[pre_shape, out_shape], tf.float32, initializer=initializer)
            bias = tf.get_variable("bias",out_shape, tf.float32, initializer=initializer)
        
        
        if activat_fn != None:
            net = activat_fn(tf.nn.xw_plus_b(inputs, weight, bias, name=name + '_out'))
        else:
            net = tf.nn.xw_plus_b(inputs, weight, bias, name=name)
        
    return net

def inception_v1(inputs, module_shape, name, flatten = False ,padding = 'SAME',initializer=tf.contrib.layers.xavier_initializer(), activat_fn=tf.nn.relu):
    
    
    
    with tf.variable_scope(name):
        
            with tf.variable_scope("1x1"):
                
                kernel_shape = module_shape[name]["1x1"]
                net1x1 = convolution_layer(inputs, [1,1,kernel_shape], [1,1,1,1], name="conv1x1", padding=padding, initializer=initializer, activat_fn = activat_fn)
                
            with tf.variable_scope("3x3"):
                
                kernel_shape = module_shape[name]["3x3"]["1x1"]
                net3x3 = convolution_layer(inputs, [1,1,kernel_shape], [1,1,1,1], name="conv1x1", padding=padding, initializer=initializer, activat_fn = activat_fn)
                kernel_shape = module_shape[name]["3x3"]["3x3"]
                net3x3 = convolution_layer(net3x3, [3,3,kernel_shape], [1,1,1,1], name="conv3x3", padding=padding, initializer=initializer, activat_fn = activat_fn)
                
            with tf.variable_scope("5x5"):
                
                kernel_shape = module_shape[name]["5x5"]["1x1"]
                net5x5 = convolution_layer(inputs, [1,1,kernel_shape], [1,1,1,1], name="conv1x1", padding=padding, initializer=initializer, activat_fn = activat_fn)
                kernel_shape = module_shape[name]["5x5"]["5x5"]
                net5x5 = convolution_layer(net5x5, [5,5,kernel_shape], [1,1,1,1], name="conv5x5", padding=padding, initializer=initializer, activat_fn = activat_fn)
                
            with tf.variable_scope("s1x1"):
                            
                kernel_shape = module_shape[name]["s1x1"]
                net_s1x1 = tf.nn.max_pool(inputs, ksize=[1, 3, 3, 1],strides=[1, 1, 1, 1], padding=padding, name = "maxpool_s1x1")
                net_s1x1 = convolution_layer(net_s1x1, [1,1,kernel_shape], [1,1,1,1], name="conv_s1x1", padding=padding, initializer=initializer, activat_fn = activat_fn)
            
            net = tf.concat([net1x1, net3x3, net5x5, net_s1x1], axis=3)
            
            if flatten == True:
                net = tf.reshape(net, [-1, int(np.prod(net.get_shape()[1:]))], name=name+"_flatout")
                
            
            return net


def shortcut(inputs, identity, name):  #Use 1X1 conv with proper stride to match dimesions
    
    in_shape =  inputs.get_shape().as_list()
    res_shape = identity.get_shape().as_list()
    
    dim_diff = [res_shape[1]/in_shape[1],
                res_shape[2]/in_shape[2]]
    
    
    if dim_diff[0] > 1  and dim_diff[1] > 1:
    
        identity = convolution_layer(identity, [1,1,in_shape[3]], [1,dim_diff[0],dim_diff[1],1], name="shotcut", padding="VALID")
    
    resout = tf.add(inputs, identity, name=name)
    
    return resout

def global_avg_pooling(inputs, flatten="False", name= 'global_avg_pooling'):
    
    in_shape =  inputs.get_shape().as_list()  
    netout = tf.nn.avg_pool(inputs, [1,in_shape[1], in_shape[2],1], [1,1,1,1],padding = 'VALID')
    
    if flatten == True:
        netout = tf.reshape(netout, [-1, int(np.prod(netout.get_shape()[1:]))], name=name+"_flatout")
        
    return netout
    
    

#def simple_shortcut_test():
#    
#    cifar10.maybe_download_and_extract()
#        
#    images_test, cls_test, labels_test = cifar10.load_test_data()
#    images_test.astype(np.float32)
#    
#    
#    model_params = {
#            
#            "conv1": [3,3, 64],
#            "conv2": [3,3, 64],
#            "conv3": [3,3,128],
#            "conv4": [3,3, 28],
#            
#            
#            }
#    
#    
#    with tf.variable_scope("test"):
#        
#        x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='x')
#        with tf.variable_scope("test"):
#            
#            x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='x')
#           
#            net1 = convolution_layer(x, model_params["conv1"], [1,1,1,1],name="conv1")
#            net1 = convolution_layer(net1, model_params["conv2"], [1,1,1,1],name="conv2")
#            net2 = convolution_layer(net1, model_params["conv4"], [1,2,2,1],name="conv3")
#            net2 = convolution_layer(net2, model_params["conv3"], [1,1,1,1],name="conv4")
#            
#            net = shortcut(net2, net1, "s1")
#            
#            net_avg = global_avg_pooling(net)
#            
#        
#        
#        
#    with tf.Session() as sess:
#        
#        sess.run(tf.global_variables_initializer())
#        
#        netout = sess.run([net, net_avg],feed_dict={x:images_test[0:5,:,:,:]})
#        
#    return netout
#
#a = simple_shortcut_test()     
#        
#
#def simple_nettest():
#
#    
#    cifar10.maybe_download_and_extract()
#    
#    images_test, cls_test, labels_test = cifar10.load_test_data()
#    images_test.astype(np.float32)
#    
#    
#    model_params = {
#            
#            "conv1": [5,5, 64],
#            "conv2": [3,3,128],
#            "inception_1":{                 
#                    "1x1":64,
#                    "3x3":{ "1x1":96,
#                            "3x3":128
#                            },
#                    "5x5":{ "1x1":16,
#                            "5x5":32
#                            },
#                    "s1x1":32
#                    },
#            "fc3": 128,
#            "fc4": 10
#            
#            }
#    
#    
#    with tf.variable_scope("test"):
#        
#        x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='x')
#       
#        net = convolution_layer(x, model_params["conv1"], [1,2,2,1],name="conv1")
#        net = convolution_layer(net, model_params["conv2"], [1,1,1,1],name="conv2", flatten=False)
#        
#        net = inception_v1(net, model_params, name= "inception_1", flatten=True)
#     
#        net = fc_layer(net, model_params["fc3"], name="fc3")
#        net = fc_layer(net, model_params["fc4"], name="fc4", activat_fn=None)
#        
#        
#    
#    with tf.Session() as sess:
#        
#        sess.run(tf.global_variables_initializer())
#        
#        
#        netout = sess.run(net, feed_dict={x:images_test[0:5,:,:,:]})
#        ut.print_operations_in_graph()
#        
#    return netout




#simpletest = simple_nettest() 

    
    
    
    


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    