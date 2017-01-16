#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 16:34:34 2016

@author: ful6ru04

This is training phase

"""

import tensorflow as tf
import os
import math

from Apps.AppIm import MyClassImages as Cim
from Apps.MyGraph import MyGraph


############################## Variables ##############################
# Learning data
data_shape = (1,150,150,1) # [batch_size,h,w,c]
learning_rate = 1e-2
imdir_pos = 'imgs/posMix'
imdir_neg = 'imgs/negAll'

# Checkpoint
step = 0 # global_step
ckpt_step = 50
ckpt_dir = './ckpt'
ckpt_name = 'net.ckpt'
epoch = 0 # !don't modify

# Summaries
logdir = "logs/"

############################## Initialize ##############################
# Training samples
ims_pos = Cim(imdir_pos,data_shape)
ims_neg = Cim(imdir_pos,data_shape)
epoch = math.ceil( (ims_pos.len()+ims_neg.len()) / data_shape[0] )
ckpt_step = epoch


# Graph variables
xs = tf.placeholder(tf.float32,data_shape,name='xs')
ys = tf.placeholder(tf.int32,  data_shape,name='ys')
train = tf.placeholder(tf.bool)
global_step = tf.Variable(0, name='global_step', trainable=False)

# Build graph
myGraph = MyGraph()
cross_entropy = myGraph.FCN(xs,train)
train_step,loss_value = myGraph.FCN_train(cross_entropy,ys,global_step)
prediction = myGraph.predict(cross_entropy)

# Run graph
sess = tf.Session()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter(logdir, sess.graph)

# Restore from checkpoint or initialize variables
saver = tf.train.Saver()
if not os.path.exists( os.path.join(ckpt_dir,'checkpoint') ):
    print('Running graph with randomly initialized weights!')
    sess.run(tf.global_variables_initializer())
    if not os.path.exists( ckpt_dir ):
        os.makedirs(ckpt_dir)
else:
    print('Running graph from last checkpoint!')
    saver.restore(sess,tf.train.latest_checkpoint(ckpt_dir))
    step = sess.run([global_step],feed_dict={})
    step = step[0]

############################## Run Graph ##############################
# Train (use "Ctrl+c" to stop in terminal)
while ( True ):
    step = step + 1
    batch_xs, batch_ys = myImages_train.nextBatch()
    
    count = 0 # use count to retrain bad samples (10 times)
    while (True):
        _, loss = sess.run([train_step, loss_value],
                             feed_dict={xs:batch_xs, ys:batch_ys, train:True})
        print( 'step ' + str(step) + ': ' + str(loss) )
    
        results = sess.run(merged,
                             feed_dict={xs:batch_xs, ys:batch_ys, train:True})
        writer.add_summary(results,step)
        writer.flush()
        
        count = count + 1
        if ( loss<retrain_loss or count>max_retrain_times ):
            break
        
    # Epoch
    if not ( step % epoch ):
        print( 'Achieved epoch ' + str(step/epoch) + '!' )
        
    # Checkpoint
    if not ( step % ckpt_step ):
        print( 'Checkpoint saved! Step ' + str(step) + '.' )
        saver.save(sess, os.path.join(ckpt_dir,ckpt_name), global_step=step)
   
sess.close()


