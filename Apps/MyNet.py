#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 12:41:42 2016

@author: ful6ru04
"""

import tensorflow as tf

class MyNet:
    
    @staticmethod
    def weight_variable( shape, name='weight', stddev=0.1 ):
        with tf.variable_scope(name+'/weights'):
            init = tf.truncated_normal(shape,stddev=stddev)
            weights = tf.Variable(init,name='ws')
            
            return weights
        
    @staticmethod
    def bias_variable( shape, name='bias' ):
        with tf.variable_scope('bias'):
            init = tf.constant(0.1,shape=shape)
            bias = tf.Variable(init,name='bs')
            
            return bias

    @staticmethod
    def conv_layer( bottom, weights, name='conv_layer' ):
        with tf.variable_scope(name):
            # pre-define bias
            if type(weights) is list:
                #conv_w = MyNet.weight_variable(weights,name=name)
                conv_w = tf.get_variable('weights', dtype=tf.float32, 
                         initializer=tf.truncated_normal(weights,stddev=0.1))
                b = MyNet.bias_variable([weights[3]],name=name)
                
                if type(weights) is tf.Variable:
                    conv_w = weights
                    b_size = weights.get_shape().as_list()[3]
                    b = MyNet.bias_variable([b_size],name=name)
                    
                # [cols,rows,channels,n]    
                conv = tf.nn.conv2d(bottom,conv_w,strides=[1,1,1,1], padding='SAME')
                return tf.nn.bias_add(conv,b)
            
    @staticmethod
    def deconv_layer(bottom,weights,shape,name='deconv_layer'): 
        #with tf.variable_scope(name):
            with tf.variable_scope(weights,reuse=True):
                deconv_w=tf.get_variable('weights')
                #deconv_w=tf.transpose(deconv_w,perm=[0,1,3,2])
                b_size = deconv_w.get_shape().as_list()[2]
                b = MyNet.bias_variable([b_size],name=name)
                deconv = tf.nn.conv2d_transpose(bottom,deconv_w,shape,strides=[1,1,1,1])
                return tf.nn.bias_add(deconv,b)
            
    @staticmethod
    def pooling_layer( bottom, name='pooling_layer' ):
        with tf.name_scope(name):
            return tf.nn.max_pool(bottom,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
    
    @staticmethod
    def relu_layer(bottom,name='relu_layer'):
        with tf.name_scope(name):    
            return tf.nn.relu(bottom)

    @staticmethod
    # bottom:[a_in,b_in]
    # upper :[b_in,c_out]
    def fully_connected(bottom,Weights,name='fc'):
        with tf.variable_scope(name):
        #with tf.name_scope(name):
            return tf.matmul(bottom,Weights)
    
    @staticmethod
    def softmax_layer(bottom,name='softmax'):
        with tf.name_scope(name):
            return tf.nn.softmax(bottom)
    
    @staticmethod
    def upsampling_layer(bottom,new_height=100,new_width=100,name='upsampling'):
        with tf.name_scope(name):
            return tf.image.resize_images(bottom, [new_height, new_width])
    
    
                
                