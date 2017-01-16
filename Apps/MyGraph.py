#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 12:41:42 2016

@author: ful6ru04
"""

import tensorflow as tf

from Apps.MyNet import MyNet as net


class MyGraph():
    
    def __init__(self):
        self.batch_size = 3
        self.height = 512
        self.width = 512
        self.channel = 1
        self.learning_rate = 1e-4
        self.categories = 2
        
        self.weights_size = {        # [k,k, in,out]
                             'conv1_1':[3,3,  1,  6],
                             'conv2_1':[3,3,  6, 16],
                             'conv3_1':[5,5, 16, 64],
                             'conv3_2':[5,5, 64,128],

                             'fc4_1'  :[1,1,128,256],
                             'fc5_1'  :[1,1,256,512],
                             'fc6_1'  :[1,1,512,128],
                                       # [k,k,out, in]
                             'deconv9_1':[3,3, 6,  2]}

    def FCN(self,xs,train):
                
        ########## LAYER DEFINITION ##########
        ### layer 1
        conv1_1 = net.conv_layer(xs, self.weights_size['conv1_1'], name='conv1_1')
        relu1_1 = net.relu_layer(conv1_1,name='relu1_1')
        relu1_1 = tf.nn.local_response_normalization(relu1_1)
        pool1_1 = net.pooling_layer(relu1_1, name='pool1_1')

        # layer 2
        conv2_1 = net.conv_layer(pool1_1, self.weights_size['conv2_1'], name='conv2_1')
        relu2_1 = net.relu_layer(conv2_1,name='relu2_1')
        relu2_1 = tf.nn.local_response_normalization(relu2_1)
        pool2_1 = net.pooling_layer(relu2_1, name='pool2_1')
        
        # layer 3
        conv3_1 = net.conv_layer(pool2_1, self.weights_size['conv3_1'], name='conv3_1')
        relu3_1 = net.relu_layer(conv3_1,name='relu3_1')
        relu3_1 = tf.nn.local_response_normalization(relu3_1)
        conv3_2 = net.conv_layer(relu3_1, self.weights_size['conv3_2'], name='conv3_2')
        relu3_2 = net.relu_layer(conv3_2,name='relu3_2')
        relu3_2 = tf.nn.local_response_normalization(relu3_2)
        
        # layer 4 (fc)
        fc4_1 = net.conv_layer(relu3_2,self.weights_size['fc4_1'], name='fc4_1')
        fc4_1 = tf.cond(train,lambda:tf.nn.dropout(fc4_1,0.5),lambda:fc4_1)
        fc4_1 = net.relu_layer(fc4_1,name='relu4_1-fc')
        fc4_1 = tf.nn.local_response_normalization(fc4_1)

        # layer 5 (fc)
        fc5_1 = net.conv_layer(fc4_1,self.weights_size['fc5_1'], name='fc5_1')
        fc5_1 = tf.cond(train,lambda:tf.nn.dropout(fc5_1,0.5),lambda:fc5_1)
        fc5_1 = net.relu_layer(fc5_1,name='relu5_1-fc')
        fc5_1 = tf.nn.local_response_normalization(fc5_1)
        
        # layer 6 (fc)
        fc6_1 = net.conv_layer(fc5_1,self.weights_size['fc6_1'], name='fc6_1')
        fc6_1 = tf.cond(train,lambda:tf.nn.dropout(fc6_1,0.5),lambda:fc6_1)
        fc6_1 = net.relu_layer(fc6_1,name='relu6_1-fc')
        fc6_1 = tf.nn.local_response_normalization(fc6_1)
        
        # upscore
        # layer 7 -> layer 3-2
        deconv7_1 = net.deconv_layer(fc6_1,'conv3_2',
                         [self.batch_size,128,128,self.weights_size['conv3_1'][3]],name='deconv7_1')
        deconv7_1_fuse = tf.add(deconv7_1,conv3_1,name='deconv7_1_fuse')
        
        # layer 7 -> layer 3-1
        deconv7_2 = net.deconv_layer(deconv7_1_fuse,'conv3_1',
                         [self.batch_size,128,128,self.weights_size['conv2_1'][3]],name='deconv7_2')
        deconv7_2_unpool = net.upsampling_layer(deconv7_2,256,256)
        deconv7_2_fuse = tf.add(deconv7_2_unpool,conv2_1,name='deconv7_2_fuse')
        
        # layer 8 -> layer 2
        deconv8_1 = net.deconv_layer(deconv7_2_fuse,'conv2_1',
                         [self.batch_size,256,256,self.weights_size['conv1_1'][3]],name='deconv8_1')
        deconv8_1_unpool = net.upsampling_layer(deconv8_1,self.height,self.width)
        deconv8_1_fuse = tf.add(deconv8_1_unpool,conv1_1,name='deconv8_1_fuse')
        
        # layer 9 -> layer 1
        deconv9_1 = net.conv_layer(deconv8_1_fuse,self.weights_size['deconv9_1'],name='deconv9_1')
        cross_entropy = deconv9_1
        # deconv9_1: [512*512*16]->[512*512*2]
        
        return cross_entropy
        
    def predict(self,cross_entropy):
        prediction = tf.argmax(cross_entropy, dimension=3)
        return prediction
        
    def FCN_train(self,cross_entropy,ys,global_step):
        
        # training solver
        with tf.name_scope('loss'):
           
            ys_reshape = tf.squeeze(ys,squeeze_dims=[3])
            
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(cross_entropy,ys_reshape)
            cross_entropy = tf.reduce_mean(cross_entropy)
            tf.summary.scalar('loss',cross_entropy)
            
        with tf.name_scope('solver'):
            train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(cross_entropy,global_step=global_step)

        return train_step, cross_entropy
    




