#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 16:34:34 2016

@author: ful6ru04

This is testing phase

"""

import tensorflow as tf
import os

import numpy as np
import scipy.misc

from Apps.MyGraph import MyGraph
from Apps.MyImages import MyImages

############################## Variables ##############################
# Graph data
data_shape = [3,512,512,1] # [batch_size,h,w,c]

# Testing variables
testing_iter = 200
output_dir = './output'
imdir_test  = './DICOM_dat4'
data_ext = 'png'

# Checkpoint
ckpt_dir = './ckpt'
ckpt_name = 'net.ckpt'

############################## Initialize ##############################
# Initialize data
myImages_test  = MyImages(imdir_test, data_shape[0],MyImages.TESTING_PHASE)

# Graph variables
xs = tf.placeholder(tf.float32,data_shape,name='xs')
ys = tf.placeholder(tf.int32,  data_shape,name='ys')
train = tf.placeholder(tf.bool)
global_step = tf.Variable(0, name='global_step', trainable=False)

# Build Graph
myGraph = MyGraph()
cross_entropy = myGraph.FCN(xs,train)
train_step,loss_value = myGraph.FCN_train(cross_entropy,ys,global_step)
prediction = myGraph.predict(cross_entropy)

# Run Graph
sess = tf.Session()

# Restore weights from checkpoint
saver = tf.train.Saver()
if not os.path.exists( os.path.join(ckpt_dir,'checkpoint') ):
    raise ValueError("Error! There is reained weights! Please train the graph first!")
    
print('Restoring graph from last checkpoint!')
saver.restore(sess,tf.train.latest_checkpoint(ckpt_dir))
step = sess.run([global_step],feed_dict={})
print('Running graph using the weight which are trained after '+
      str(step[0])+' iteration training phase!')

############################## Run Graph ##############################
# Make output directory
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for step in range(testing_iter):

    print( 'Testing step: ' + str(step) )
    batch_xs = myImages_test.nextBatch()
    pred = sess.run(prediction,feed_dict={xs:batch_xs, train:False})
    
    for batch in range(data_shape[0]):
        img = np.squeeze(batch_xs[batch,:,:])
        scipy.misc.imsave(output_dir+'/test'+str(step*data_shape[0]+0)+'_in'+'.jpg', img)
        
        pr = np.squeeze(pred[batch,:,:])
        scipy.misc.imsave(output_dir+'/test'+str(step*data_shape[0]+0)+'_out'+'.jpg', pr)

sess.close()

