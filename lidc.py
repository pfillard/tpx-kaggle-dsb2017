# Copyright (C) 2017 Therapixel / Pierre Fillard (pfillard@therapixel.com)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import math
import glob
import h5py as h5
import numpy as np
import scipy as sp
from scipy import ndimage
from skimage import morphology
from skimage.feature import peak_local_max
import time
from time import sleep
import sys
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, convolution2d, flatten, batch_norm, max_pool2d, dropout, l2_regularizer
from tensorflow.python.ops.nn import relu, elu, relu6, sigmoid, tanh, softmax, relu1, avg_pool3d
from tensorflow.python.ops import variable_scope
import SimpleITK as sitk


MOVING_AVERAGE_DECAY = 0.9
PATCH_SIZE = 48
L2_REGULARIZER=1e-5


def _variable_on_cpu(name, shape, initializer, regularizer=None):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer, regularizer=regularizer, dtype=tf.float32)
    return var

    
def _variable_with_weight_decay(name, shape, initializer, regularizer=l2_regularizer(L2_REGULARIZER)):
    var = _variable_on_cpu(
        name,
        shape,
        initializer,
        regularizer)
    return var

    
def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            if g is None:
                break
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        if (len(grads)==0):
            continue
            
        grad = tf.concat(grads, axis=0)
        grad = tf.reduce_mean(grad, axis=0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

    
def convolution3d(input, num_outputs, kernel_size, strides, padding, activation_fn, is_training, 
                  weights_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.zeros_initializer(), 
                  with_bias=True, with_bn=False, with_bn_in_place=True, with_bn_after_relu=False, scope=None):
    with tf.variable_scope(scope):
        num_input_channels = input.get_shape()[4]
        weights_shape = (list(kernel_size) + [num_input_channels, num_outputs])
        weights = _variable_with_weight_decay(name='weights',
                                              shape=weights_shape,
                                              initializer=weights_initializer)        
        if with_bn and not with_bn_after_relu:
            if with_bn_in_place:
                input = batch_norm(input, decay=0.9, activation_fn=None, is_training=is_training, center=True, scale=False,
                                   updates_collections=None, scope='batch_norm')
            else:
                input = batch_norm(input, decay=0.9, activation_fn=None, is_training=is_training, center=True, scale=False,
                                   scope='batch_norm')
        output = tf.nn.conv3d(input, filter=weights, strides=strides, padding=padding)
        if with_bias:
            bias = _variable_with_weight_decay(name='bias',
                                               shape=[num_outputs],
                                               initializer=bias_initializer,
                                               regularizer=None
                                              )
            output = tf.nn.bias_add(output, bias)
        if (activation_fn is not None):                        
            output = activation_fn(output, name='relu')
            if with_bn and with_bn_after_relu:
                if with_bn_in_place:
                    output = batch_norm(output, decay=0.9, activation_fn=None, is_training=is_training, center=True, scale=False,
                                        updates_collections=None, scope='batch_norm')
                else:
                    output = batch_norm(output, decay=0.9, activation_fn=None, is_training=is_training, center=True, scale=False,
                                        scope='batch_norm')
    return output

    
def _create_base_network(input, is_training, with_bn=False):
    input = convolution3d(input, num_outputs=16, kernel_size=[1,1,1], strides=[1,1,1,1,1], padding='VALID', 
                          activation_fn=relu1, 
                          is_training=is_training, 
                          weights_initializer=tf.random_normal_initializer(1.0, stddev=0.1), 
                          bias_initializer=tf.zeros_initializer(),
                          with_bias=True,
                          with_bn=False,
                          scope='conv01')
    
    # tower 1 conv 1 (32 filters) + max pool + drop out    
    input = convolution3d(input, num_outputs=64, kernel_size=[3,3,3], strides=[1,1,1,1,1], padding='VALID', 
                          activation_fn=relu, is_training=is_training, with_bias=True, with_bn=with_bn, 
                          with_bn_in_place=True, scope='conv11')
    input = tf.nn.max_pool3d(input, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='VALID', name="pool11")
    
    # tower 1 conv 2 (64 filters) + max pool + drop out
    input = convolution3d(input, num_outputs=128, kernel_size=[3,3,3], strides=[1,1,1,1,1], padding='VALID', 
                          activation_fn=relu, is_training=is_training, with_bias=True, with_bn=with_bn, 
                          with_bn_in_place=True, scope='conv31')    
    input = tf.nn.max_pool3d(input, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='VALID', name="pool31")    

    # tower 1 conv 3 (128 filters) + max pool + drop out
    input = convolution3d(input, num_outputs=256, kernel_size=[3,3,3], strides=[1,1,1,1,1], padding='VALID', 
                          activation_fn=relu, is_training=is_training, with_bias=True, with_bn=with_bn, 
                          with_bn_in_place=True, scope='conv41')
    input = convolution3d(input, num_outputs=256, kernel_size=[3,3,3], strides=[1,1,1,1,1], padding='VALID', 
                          activation_fn=relu, is_training=is_training, with_bias=True, with_bn=with_bn, 
                          with_bn_in_place=True, scope='conv41_1')
    input = tf.nn.max_pool3d(input, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='VALID', name="pool41")

    # tower 1 conv 5 (512 filters) -> fully connected
    input = convolution3d(input, num_outputs=512, kernel_size=[3,3,3], strides=[1,1,1,1,1], padding='VALID',
                          activation_fn=relu,is_training=is_training, with_bias=True, with_bn=with_bn,
                          with_bn_in_place=True, scope='conv61')
    input = dropout(input, keep_prob=0.5, is_training=is_training, scope="dropout61")
    input = convolution3d(input, num_outputs=512, kernel_size=[1,1,1], strides=[1,1,1,1,1], padding='VALID',
                          activation_fn=relu1, is_training=is_training, with_bias=True, with_bn=with_bn, 
                          with_bn_in_place=True, scope='conv71')
    input = dropout(input, keep_prob=0.5, is_training=is_training, scope="dropout71")
    
    return input

    
def _create_base_network_all_scores(input, is_training, padding='VALID'):
    with_bn = True
    input = batch_norm(input, decay=0.9, activation_fn=None, is_training=is_training, center=True, scale=False,
                       updates_collections=None, scope='batch_norm_0')
    
    input = convolution3d(input, num_outputs=16, kernel_size=[3,3,3], strides=[1,1,1,1,1], padding=padding,
                          activation_fn=relu, is_training=is_training, with_bias=False, with_bn=with_bn, 
                          with_bn_after_relu=True, scope='conv1')
    input = tf.nn.max_pool3d(input, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding=padding, name="pool1")
    
    # tower 1 conv 3 (64 filters) + max pool + drop out
    input = convolution3d(input, num_outputs=32, kernel_size=[3,3,3], strides=[1,1,1,1,1], padding=padding, 
                          activation_fn=relu, is_training=is_training, with_bias=False, with_bn=with_bn, 
                          with_bn_after_relu=True, scope='conv2')
    input = tf.nn.max_pool3d(input, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding=padding, name="pool2")
    
    # tower 1 conv 3 (128 filters) + max pool + drop out
    input = convolution3d(input, num_outputs=64, kernel_size=[3,3,3], strides=[1,1,1,1,1], padding=padding, 
                          activation_fn=relu, is_training=is_training, with_bias=False, with_bn=with_bn, 
                          with_bn_after_relu=True, scope='conv3_1')
    input = convolution3d(input, num_outputs=64, kernel_size=[3,3,3], strides=[1,1,1,1,1], padding=padding, 
                          activation_fn=relu, is_training=is_training, with_bias=False, with_bn=with_bn, 
                          with_bn_after_relu=True, scope='conv3_2')
    input = tf.nn.max_pool3d(input, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding=padding, name="pool3")
    
    # tower 1 conv 3 (128 filters) + max pool + drop out
    input = convolution3d(input, num_outputs=128, kernel_size=[3,3,3], strides=[1,1,1,1,1], padding=padding, 
                          activation_fn=relu, is_training=is_training, with_bias=False, with_bn=with_bn, 
                          with_bn_after_relu=True, scope='conv4_1')
    input = convolution3d(input, num_outputs=128, kernel_size=[3,3,3], strides=[1,1,1,1,1], padding=padding, 
                          activation_fn=relu, is_training=is_training, with_bias=False, with_bn=with_bn, 
                          with_bn_after_relu=True, scope='conv4_2')
    input = tf.nn.max_pool3d(input, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding=padding, name="pool4")
            
    # FC layers as convolutional
    input = dropout(input, keep_prob=0.5, is_training=is_training, scope="dropout_0")
    input = convolution3d(input, num_outputs=256, kernel_size=[4,4,4], strides=[1,1,1,1,1], padding='VALID', 
                          activation_fn=relu, is_training=is_training, with_bias=False, with_bn=True, 
                          with_bn_after_relu=True, scope='conv6_1')
    input = dropout(input, keep_prob=0.5, is_training=is_training, scope="dropout_1")
    input = convolution3d(input, num_outputs=256, kernel_size=[1,1,1], strides=[1,1,1,1,1], padding='VALID', 
                          activation_fn=None, is_training=is_training, with_bias=False, with_bn=False,  # false on purpose
                          with_bn_after_relu=True, scope='conv6_2')    
    
    return input

    
def _create_base_network_emphyseme(input, is_training):
    with_bn = True
    padding='VALID'
    
    input = batch_norm(input, decay=0.9, activation_fn=None, is_training=is_training, center=True, scale=False,
                       updates_collections=None, scope='batch_norm_0')
    
    input = convolution3d(input, num_outputs=16, kernel_size=[3,3,3], strides=[1,1,1,1,1], padding=padding,
                          activation_fn=relu, is_training=is_training, with_bias=False, with_bn=with_bn, 
                          with_bn_after_relu=True, scope='conv1')
    input = tf.nn.max_pool3d(input, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding=padding, name="pool1")
    
    # tower 1 conv 3 (64 filters) + max pool + drop out
    input = convolution3d(input, num_outputs=32, kernel_size=[3,3,3], strides=[1,1,1,1,1], padding=padding, 
                          activation_fn=relu, is_training=is_training, with_bias=False, with_bn=with_bn, 
                          with_bn_after_relu=True, scope='conv2')
    input = tf.nn.max_pool3d(input, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding=padding, name="pool2")
    
    # tower 1 conv 3 (128 filters) + max pool + drop out
    input = convolution3d(input, num_outputs=64, kernel_size=[3,3,3], strides=[1,1,1,1,1], padding=padding, 
                          activation_fn=relu, is_training=is_training, with_bias=False, with_bn=with_bn, 
                          with_bn_after_relu=True, scope='conv3_1')
    input = convolution3d(input, num_outputs=64, kernel_size=[3,3,3], strides=[1,1,1,1,1], padding=padding, 
                          activation_fn=relu, is_training=is_training, with_bias=False, with_bn=with_bn, 
                          with_bn_after_relu=True, scope='conv3_2')
    input = tf.nn.max_pool3d(input, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding=padding, name="pool3")   
            
    # FC layers as convolutional
    input = convolution3d(input, num_outputs=256, kernel_size=[5,5,5], strides=[1,1,1,1,1], padding='VALID', 
                          activation_fn=relu, is_training=is_training, with_bias=False, with_bn=True, 
                          with_bn_after_relu=True, scope='conv6_1')
    input = dropout(input, keep_prob=0.5, is_training=is_training, scope="dropout_1")
    input = convolution3d(input, num_outputs=256, kernel_size=[1,1,1], strides=[1,1,1,1,1], padding='VALID', 
                          activation_fn=relu, is_training=is_training, with_bias=False, with_bn=False,  # false on purpose
                          with_bn_after_relu=True, scope='conv6_2')    
    
    return input   

    
def inference(input, is_training, with_bn=False):
    input = _create_base_network(input, is_training=is_training, with_bn=with_bn)
        
    logits = convolution3d(input, num_outputs=2, kernel_size=[1,1,1], strides=[1,1,1,1,1], padding='VALID',
                           activation_fn=None, is_training=is_training, with_bias=True, with_bn=with_bn, 
                           with_bn_in_place=True, scope='conv81')
    
    return tf.squeeze(logits)
       
    
def inference_all_scores(input, num_outputs, is_training):
    features = _create_base_network_all_scores(input, is_training=is_training, padding='SAME')    
    input = tf.nn.relu(features)
    with tf.variable_scope('final'):
        input_bn = batch_norm(input, decay=0.9, activation_fn=None, is_training=is_training, center=True, 
                             scale=False, updates_collections=None, scope='batch_norm')
        if num_outputs==9:
            output = []
            for i in range(num_outputs):
                preds = dropout(input_bn, keep_prob=0.5, is_training=is_training, scope="dropout_%d"%i)
                preds = convolution3d(preds, num_outputs=1, kernel_size=[1,1,1], strides=[1,1,1,1,1], padding='VALID',
                                      activation_fn=None, is_training=is_training, with_bias=True, with_bn=False,
                                      scope='final_fc_%d'%i)
                output.append(preds)
            preds = tf.concat(output, axis=4)            
        else:
            #preds = dropout(input_bn, keep_prob=0.5, is_training=is_training, scope='dropout')
            preds = convolution3d(input_bn, num_outputs=num_outputs, kernel_size=[1,1,1], strides=[1,1,1,1,1], padding='VALID',
                                  activation_fn=None, is_training=is_training, with_bias=True, with_bn=False,
                                  scope='final_fc')            
    return preds, features

    
def inference_emphyseme(input, is_training, num_outputs=3):
    input = _create_base_network_emphyseme(input, is_training=is_training)
    with tf.variable_scope('final'):
        input = dropout(input, keep_prob=0.5, is_training=is_training, scope='dropout')
        input = convolution3d(input, num_outputs=num_outputs, kernel_size=[1,1,1], strides=[1,1,1,1,1], padding='VALID',
                              activation_fn=None, is_training=is_training, with_bias=True, with_bn=False, scope='final_fc')
                    
    return input

    

def loss(logits, labels, with_regularization=False):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross_entropy')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')
    if with_regularization:
        reg_vars = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        l2_loss_term = tf.add_n([v for v in reg_vars])
        cross_entropy_mean = cross_entropy_mean + l2_loss_term
    return cross_entropy_mean

    
def l2_loss(preds, values, with_regularization=False):
    l2_loss = tf.square(preds-values, name='square')
    l2_loss_mean = tf.reduce_mean(l2_loss, name='cross_entropy_mean')
    if with_regularization:
        reg_vars = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        l2_loss_term = tf.add_n([v for v in reg_vars])
        l2_loss_mean = l2_loss_mean + l2_loss_term
    return l2_loss_mean

    
def center_of_mass(mask):
    com=[0.,0.,0.]
    total = 0.
    a = np.nonzero(mask)
    for i in range(len(a[0])):
        z,y,x=a[0][i],a[1][i],a[2][i]
        b = mask[z,y,x]
        com += np.array([z,y,x])*b
        total += b
    if (total>0.):
        com /= total
    return np.round(com).astype(np.int32)

    
# scan a volume given a model and returns a score map
def scan_volume(volume, model, with_bn=False):
    a=8 #downsampling factor of the nework due to the pooling operations
    b=48 #patch size / receptive field of the netword
    img_z, img_y, img_x = volume.shape
    
    score_map = np.zeros(shape=((img_z-b)//a+1, (img_y-b)//a+1, (img_x-b)//a+1), dtype=np.float32)
    
    # reset graph first
    tf.reset_default_graph()        
    
    # create new graph to test model
    with tf.Graph().as_default(), tf.device('/gpu:0'):

        # Setting up placeholder, this is where your data enters the graph!
        x_pl = tf.placeholder(tf.float32, shape=(1, None, None, None, 1), name='data_x')

        logits = inference(x_pl, is_training=False, with_bn=with_bn)
        y = tf.nn.softmax(logits)
        
        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
    
        # restricting memory usage, TensorFlow is greedy and will use all memory otherwise
        gpu_opts = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    
        # initialize the Session
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_opts, 
                                                allow_soft_placement=True, 
                                                log_device_placement=True))
        sess.run(tf.global_variables_initializer())
        
        print('restoring model', model)
        saver.restore(sess, model)

    # scanning loop
    for k in range(score_map.shape[0]):
        k_start = k*a
        k_end = k*a+b
        region=volume[k_start:k_end, :, :].reshape(1, k_end-k_start, img_y, img_x, 1)
        
        # divide in 4?
        mid_height = int(b*math.ceil(float(img_y//2)/b))
        mid_width  = int(b*math.ceil(float(img_x//2)/b))
        region_top_left = region[:, :, 0:mid_height, 0:mid_width, :]
        region_top_right = region[:, :, 0:mid_height, mid_width-b:img_x, :]
        region_bottom_left = region[:, :, mid_height-b:img_y, 0:mid_width, :]
        region_bottom_right = region[:, :, mid_height-b:img_y, mid_width-b:img_x, :]
        
        feed_dict_vol = {x_pl: region_top_left}
        fetches = [y]
        res = sess.run(fetches=fetches, feed_dict=feed_dict_vol)        
        pred = res[0]
        r = score_map[k,:(mid_height-b)//a+1,:(mid_width-b)//a+1]
        pred = pred.reshape(tuple(r.shape) + tuple([2]))
        score_map[k,:(mid_height-b)//a+1,:(mid_width-b)//a+1]=pred[:,:,1]
        
        feed_dict_vol = {x_pl: region_top_right}
        fetches = [y]
        res = sess.run(fetches=fetches, feed_dict=feed_dict_vol)        
        pred = res[0]
        r = score_map[k,:(mid_height-b)//a+1,(mid_width-b)//a:]
        pred = pred.reshape(tuple(r.shape) + tuple([2]))
        score_map[k,:(mid_height-b)//a+1,(mid_width-b)//a:]=pred[:,:,1]
        
        feed_dict_vol = {x_pl: region_bottom_left}
        fetches = [y]
        res = sess.run(fetches=fetches, feed_dict=feed_dict_vol)        
        pred = res[0]
        r = score_map[k,(mid_height-b)//a:,:(mid_width-b)//a+1]
        pred = pred.reshape(tuple(r.shape) + tuple([2]))
        score_map[k,(mid_height-b)//a:,:(mid_width-b)//a+1]=pred[:,:,1]
        
        feed_dict_vol = {x_pl: region_bottom_right}
        fetches = [y]
        res = sess.run(fetches=fetches, feed_dict=feed_dict_vol)        
        pred = res[0]
        r = score_map[k,(mid_height-b)//a:,(mid_width-b)//a:]
        pred = pred.reshape(tuple(r.shape) + tuple([2]))
        score_map[k,(mid_height-b)//a:,(mid_width-b)//a:]=pred[:,:,1]
    sess.close()
    
    # upscale score map to match input size
    score_map = sp.ndimage.zoom(score_map, a, order=1) # zoom by factor a
    offset = (b-a)//2
    score_map = np.pad(score_map, ((offset,offset),(offset,offset),(offset,offset)), # pad to center
                       'constant', constant_values=((0, 0),(0, 0),(0, 0)))    
    
    return score_map
    
    
# scan a volume given a model and returns a score map
def scan_volume_emphyseme(volume, model):
    a=8 #downsampling factor of the nework due to the pooling operations
    b=64 #patch size
    img_z, img_y, img_x = volume.shape
    
    score_map_0 = np.zeros(shape=((img_z-b)//a+1, (img_y-b)//a+1, (img_x-b)//a+1), dtype=np.float32)
    score_map_1 = np.zeros(shape=((img_z-b)//a+1, (img_y-b)//a+1, (img_x-b)//a+1), dtype=np.float32)
    score_map_2 = np.zeros(shape=((img_z-b)//a+1, (img_y-b)//a+1, (img_x-b)//a+1), dtype=np.float32)
    
    # reset graph first
    tf.reset_default_graph()        
    
    # create new graph to test model
    with tf.Graph().as_default(), tf.device('/gpu:0'):

        # Setting up placeholder, this is where your data enters the graph!
        x_pl = tf.placeholder(tf.float32, shape=(1, None, None, None, 1), name='data_x')

        logits = inference_emphyseme(x_pl, is_training=False)
        y = tf.nn.softmax(logits)
        
        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
    
        # restricting memory usage, TensorFlow is greedy and will use all memory otherwise
        gpu_opts = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    
        # initialize the Session
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_opts, 
                                                allow_soft_placement=True, 
                                                log_device_placement=True))
        sess.run(tf.global_variables_initializer())
        
        print('restoring model', model)
        saver.restore(sess, model)

    # scanning loop
    for k in range(score_map_0.shape[0]):
        k_start = k*a
        k_end = k*a+b
        region=volume[k_start:k_end, :, :].reshape(1, k_end-k_start, img_y, img_x, 1)
        
        # divide in 4?
        mid_height = int(b*math.ceil(float(img_y//2)/b))
        mid_width  = int(b*math.ceil(float(img_x//2)/b))
        region_top_left = region[:, :, 0:mid_height, 0:mid_width, :]
        region_top_right = region[:, :, 0:mid_height, mid_width-b:img_x, :]
        region_bottom_left = region[:, :, mid_height-b:img_y, 0:mid_width, :]
        region_bottom_right = region[:, :, mid_height-b:img_y, mid_width-b:img_x, :]
        
        feed_dict_vol = {x_pl: region_top_left}
        fetches = [y]
        res = sess.run(fetches=fetches, feed_dict=feed_dict_vol)        
        pred = res[0]
        r = score_map_0[k,:(mid_height-b)//a+1,:(mid_width-b)//a+1]
        pred = pred.reshape(tuple(r.shape) + tuple([3]))
        score_map_0[k,:(mid_height-b)//a+1,:(mid_width-b)//a+1]=pred[:,:,0]
        score_map_1[k,:(mid_height-b)//a+1,:(mid_width-b)//a+1]=pred[:,:,1]
        score_map_2[k,:(mid_height-b)//a+1,:(mid_width-b)//a+1]=pred[:,:,2]
        
        feed_dict_vol = {x_pl: region_top_right}
        fetches = [y]
        res = sess.run(fetches=fetches, feed_dict=feed_dict_vol)        
        pred = res[0]
        r = score_map_0[k,:(mid_height-b)//a+1,(mid_width-b)//a:]
        pred = pred.reshape(tuple(r.shape) + tuple([3]))        
        score_map_0[k,:(mid_height-b)//a+1,(mid_width-b)//a:]=pred[:,:,0]
        score_map_1[k,:(mid_height-b)//a+1,(mid_width-b)//a:]=pred[:,:,1]
        score_map_2[k,:(mid_height-b)//a+1,(mid_width-b)//a:]=pred[:,:,2]
        
        feed_dict_vol = {x_pl: region_bottom_left}
        fetches = [y]
        res = sess.run(fetches=fetches, feed_dict=feed_dict_vol)        
        pred = res[0]
        r = score_map_0[k,(mid_height-b)//a:,:(mid_width-b)//a+1]
        pred = pred.reshape(tuple(r.shape) + tuple([3]))
        score_map_0[k,(mid_height-b)//a:,:(mid_width-b)//a+1]=pred[:,:,0]
        score_map_1[k,(mid_height-b)//a:,:(mid_width-b)//a+1]=pred[:,:,1]
        score_map_2[k,(mid_height-b)//a:,:(mid_width-b)//a+1]=pred[:,:,2]
        
        feed_dict_vol = {x_pl: region_bottom_right}
        fetches = [y]
        res = sess.run(fetches=fetches, feed_dict=feed_dict_vol)        
        pred = res[0]
        r = score_map_0[k,(mid_height-b)//a:,(mid_width-b)//a:]
        pred = pred.reshape(tuple(r.shape) + tuple([3]))
        score_map_0[k,(mid_height-b)//a:,(mid_width-b)//a:]=pred[:,:,0]
        score_map_1[k,(mid_height-b)//a:,(mid_width-b)//a:]=pred[:,:,1]
        score_map_2[k,(mid_height-b)//a:,(mid_width-b)//a:]=pred[:,:,2]
    sess.close()
    
    # upscale score map to match input size
    score_map_0 = sp.ndimage.zoom(score_map_0, a, order=1) # zoom by factor a
    score_map_1 = sp.ndimage.zoom(score_map_1, a, order=1) # zoom by factor a
    score_map_2 = sp.ndimage.zoom(score_map_2, a, order=1) # zoom by factor a
    offset = (b-a)//2
    score_map_0 = np.pad(score_map_0, ((offset,offset),(offset,offset),(offset,offset)), # pad to center
                       'constant', constant_values=((0, 0),(0, 0),(0, 0)))    
    score_map_1 = np.pad(score_map_1, ((offset,offset),(offset,offset),(offset,offset)), # pad to center
                       'constant', constant_values=((0, 0),(0, 0),(0, 0)))    
    score_map_2 = np.pad(score_map_2, ((offset,offset),(offset,offset),(offset,offset)), # pad to center
                       'constant', constant_values=((0, 0),(0, 0),(0, 0)))    
    
    return score_map_0, score_map_1, score_map_2

        
# scan a volume given a model and returns a score map
def scan_volume_lung_segmentation(volume, model):
    a=8 #downsampling factor of the nework due to the pooling operations
    b=64 #patch size
    img_z, img_y, img_x = volume.shape
    
    score_map = np.zeros(shape=((img_z-b)//a+1, (img_y-b)//a+1, (img_x-b)//a+1), dtype=np.float32)
    
    # reset graph first
    tf.reset_default_graph()        
    
    # create new graph to test model
    with tf.Graph().as_default(), tf.device('/gpu:0'):

        # Setting up placeholder, this is where your data enters the graph!
        x_pl = tf.placeholder(tf.float32, shape=(1, None, None, None, 1), name='data_x')

        logits = inference_emphyseme(x_pl, is_training=False, num_outputs=2)
        y = tf.nn.softmax(logits)
        
        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
    
        # restricting memory usage, TensorFlow is greedy and will use all memory otherwise
        gpu_opts = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    
        # initialize the Session
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_opts, 
                                                allow_soft_placement=True, 
                                                log_device_placement=True))
        sess.run(tf.global_variables_initializer())
        
        print('restoring model', model)
        saver.restore(sess, model)

    # scanning loop
    for k in range(score_map.shape[0]):
        k_start = k*a
        k_end = k*a+b
        region=volume[k_start:k_end, :, :].reshape(1, k_end-k_start, img_y, img_x, 1)
        
        # divide in 4?
        mid_height = int(b*math.ceil(float(img_y//2)/b))
        mid_width  = int(b*math.ceil(float(img_x//2)/b))
        region_top_left = region[:, :, 0:mid_height, 0:mid_width, :]
        region_top_right = region[:, :, 0:mid_height, mid_width-b:img_x, :]
        region_bottom_left = region[:, :, mid_height-b:img_y, 0:mid_width, :]
        region_bottom_right = region[:, :, mid_height-b:img_y, mid_width-b:img_x, :]
        
        feed_dict_vol = {x_pl: region_top_left}
        fetches = [y]
        res = sess.run(fetches=fetches, feed_dict=feed_dict_vol)        
        pred = res[0]
        r = score_map[k,:(mid_height-b)//a+1,:(mid_width-b)//a+1]
        pred = pred.reshape(tuple(r.shape) + tuple([2]))
        score_map[k,:(mid_height-b)//a+1,:(mid_width-b)//a+1]=pred[:,:,1]
        
        feed_dict_vol = {x_pl: region_top_right}
        fetches = [y]
        res = sess.run(fetches=fetches, feed_dict=feed_dict_vol)        
        pred = res[0]
        r = score_map[k,:(mid_height-b)//a+1,(mid_width-b)//a:]
        pred = pred.reshape(tuple(r.shape) + tuple([2]))
        score_map[k,:(mid_height-b)//a+1,(mid_width-b)//a:]=pred[:,:,1]
        
        feed_dict_vol = {x_pl: region_bottom_left}
        fetches = [y]
        res = sess.run(fetches=fetches, feed_dict=feed_dict_vol)        
        pred = res[0]
        r = score_map[k,(mid_height-b)//a:,:(mid_width-b)//a+1]
        pred = pred.reshape(tuple(r.shape) + tuple([2]))
        score_map[k,(mid_height-b)//a:,:(mid_width-b)//a+1]=pred[:,:,1]
        
        feed_dict_vol = {x_pl: region_bottom_right}
        fetches = [y]
        res = sess.run(fetches=fetches, feed_dict=feed_dict_vol)        
        pred = res[0]
        r = score_map[k,(mid_height-b)//a:,(mid_width-b)//a:]
        pred = pred.reshape(tuple(r.shape) + tuple([2]))
        score_map[k,(mid_height-b)//a:,(mid_width-b)//a:]=pred[:,:,1]
    sess.close()
    
    # upscale score map to match input size
    score_map = sp.ndimage.zoom(score_map, a, order=1) # zoom by factor a
    offset = (b-a)//2
    score_map = np.pad(score_map, ((offset,offset),(offset,offset),(offset,offset)), # pad to center
                       'constant', constant_values=((0, 0),(0, 0),(0, 0)))    
    
    return score_map

    
def refine_scoremap(volume, positions, radius, model_list, max_ite=3, with_bn=False):
    offset = 24
    volume_padded = np.pad(volume, ((offset,offset),(offset,offset),(offset,offset)), # pad to center
                    'constant', constant_values=((0, 0),(0, 0),(0, 0)))    
    
    # reset graph first
    tf.reset_default_graph()        
    
    # create new graph to test model
    with tf.Graph().as_default(), tf.device('/gpu:0'):

        # Setting up placeholder, this is where your data enters the graph!
        x_pl = tf.placeholder(tf.float32, shape=(None, 48, 48, 48, 1), name='data_x')

        with tf.name_scope('tower_0') as scope:                                        
            logits = inference(x_pl, is_training=False, with_bn=with_bn)
            y = tf.nn.softmax(logits)
        
        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
    
        # restricting memory usage, TensorFlow is greedy and will use all memory otherwise
        gpu_opts = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    
        # initialize the Session
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_opts, 
                                                allow_soft_placement=True, 
                                                log_device_placement=True))
        sess.run(tf.global_variables_initializer())
    
    img_z, img_y, img_x = volume.shape    
    patches = np.zeros(shape=(64,48,48,48), dtype=np.float32)
    s = np.zeros(shape=(8,8,8))
    diameter = radius * 2
    step = diameter // 8
    candidates = []
    score_map = np.zeros(shape=volume.shape, dtype=np.float32)
    for p in range(len(positions)):
        pk,pj,pi = positions[p]
        print('processing %d/%d'%(p+1,len(positions)),pk,pj,pi)        
        score_mask = np.zeros(shape=(volume.shape), dtype=np.float32)        
        history = []
        for ite in range(max_ite):
            kmin=max(0,pk-radius)
            kmax=min(pk+radius,img_z)
            jmin=max(0,pj-radius)
            jmax=min(pj+radius,img_y)
            imin=max(0,pi-radius)
            imax=min(pi+radius,img_x)
            score_patch = np.zeros(shape=(kmax-kmin,jmax-jmin,imax-imin), dtype=np.float32)
            for prev_model in model_list:
                saver.restore(sess, prev_model)
                for k in range(kmin,kmax,step):
                    index=0
                    for j in range(jmin,jmax,step):
                        for i in range(imin,imax,step):
                            patches[index] = volume_padded[k:k+48,j:j+48,i:i+48]
                            index = index+1
                    feed_dict_vol = {
                        x_pl: patches.reshape(64,48,48,48,1),
                    }
                    fetches = [y]
                    res = sess.run(fetches=fetches, feed_dict=feed_dict_vol)
                    s[(k-kmin)//step] = res[0][:,1].reshape((jmax-jmin)//step,(imax-imin)//step)
                if (step>1):
                    score_patch += sp.ndimage.zoom(s, step, order=1).reshape(kmax-kmin,jmax-jmin,imax-imin)
                else:
                    score_patch += s                    
            score_patch /= len(model_list)
            score_map[kmin:kmax,jmin:jmax,imin:imax] = score_patch
            score_mask[kmin:kmax,jmin:jmax,imin:imax] = 1.
            u=score_patch.ravel()
            u=np.sort(u)
            t=u[-len(u)//10]
            npk1,npj1,npi1=center_of_mass(score_map*score_mask*(score_map>=t))
            npk2,npj2,npi2=center_of_mass(volume*score_mask*(score_map>=t))
            a1=score_map[npk1,npj1,npi1]
            a2=score_map[npk2,npj2,npi2]
            alpha1=a1*(1-a2)/(1+a2)
            alpha2=a2/(1+a1)
            npk=int(round((alpha1*npk1+alpha2*npk2)/(alpha1+alpha2)))
            npj=int(round((alpha1*npj1+alpha2*npj2)/(alpha1+alpha2)))
            npi=int(round((alpha1*npi1+alpha2*npi2)/(alpha1+alpha2)))
            if (npk==pk and npj==pj and npi==pi):
                break
            history_found = False
            for h in history:
                if (h[0]==npk and h[1]==npj and h[2]==npi):
                    history_found = True
                    break
            if history_found:
                break
            pk=npk
            pj=npj
            pi=npi
            history.append([pk, pj, pi])
        candidates.append([pk,pj,pi])
    sess.close()
    
    return score_map, candidates


def estimate_scores(patches, num_outputs, model, batch_size=8):
    patch_size = 64
    num_features = 256
    
    # reset graph first
    tf.reset_default_graph()        
    
    # create new graph to test model
    with tf.Graph().as_default(), tf.device('/gpu:0'):

        # Setting up placeholder, this is where your data enters the graph!
        x_pl = tf.placeholder(tf.float32, shape=(None, patch_size, patch_size, patch_size, 1), name='data_x')

        with tf.name_scope('tower_0') as scope:                                        
            y, f = inference_all_scores(x_pl, num_outputs=num_outputs, is_training=False)
            y = tf.reshape(y, [-1,num_outputs])
            f = tf.reshape(f, [-1,num_features])
        
        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
    
        # restricting memory usage, TensorFlow is greedy and will use all memory otherwise
        gpu_opts = tf.GPUOptions(allow_growth=True)
    
        # initialize the Session
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_opts, 
                                                allow_soft_placement=True, 
                                                log_device_placement=True))
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, model)
                
    fetches = [y, f]
    patch_count = patches.shape[0]
    
    scores = np.zeros(shape=(patch_count, num_outputs), dtype=np.float32)
    features = np.zeros(shape=(patch_count, num_features), dtype=np.float32)
    
    for i in range(0,patch_count,batch_size):
        start_index=i
        stop_index = min(i+batch_size,patch_count)
        
        p = patches[start_index:stop_index]
        feed_dict_vol = {
            x_pl: p.reshape(tuple(p.shape) + tuple([1]))
        }
        res = sess.run(fetches=fetches, feed_dict=feed_dict_vol)
        scores[start_index:stop_index] = res[0]
        features[start_index:stop_index] = res[1]
    sess.close()
        
    return scores, features


def estimate_scores_from_coordinates(coordinates, series_filename, num_outputs, model, target_spacing=[0.625,0.625,0.625], batch_size=8):    
    patch_size = 64
    offset = patch_size//2
    
    spacing_ratio = target_spacing[0] / 0.625
    
    itk_image = load_itk_image(series_filename)
    volume, origin, spacing, orientation = parse_itk_image(itk_image)
    
    if ~(spacing==target_spacing).all():        
        # resample using itk
        print('resampling itk volume')
        padding_value = volume.min()
        img_z, img_y, img_x = volume.shape
        img_z_new = int(np.round(img_z*spacing[2]/target_spacing[2]))
        img_y_new = int(np.round(img_y*spacing[1]/target_spacing[1]))
        img_x_new = int(np.round(img_x*spacing[0]/target_spacing[0]))

        itk_image = resample_itk_image(itk_image, [img_x_new,img_y_new,img_z_new], target_spacing, int(padding_value))
        volume, origin, spacing, orientation = parse_itk_image(itk_image)
            
    volume = volume.astype(np.float32)
    volume = normalizePlanes(volume)
    
    volume = np.pad(volume, ((offset,offset),(offset,offset),(offset,offset)), # pad to center
                             'constant', constant_values=((0, 0),(0, 0),(0, 0)))    
    
    patches = np.zeros(shape=(len(coordinates),patch_size,patch_size,patch_size), dtype=np.float32)
    
    for i in range(len(coordinates)):
        wc = coordinates[i][0:3]
        ic = worldToVoxelCoord(wc, origin=origin, spacing=spacing, orientation=orientation)
        kk,jj,ii = ic
        kk = int(round(kk))
        jj = int(round(jj))
        ii = int(round(ii))
        patches[i]=volume[kk:kk+patch_size,jj:jj+patch_size,ii:ii+patch_size]
    
    scores, features = estimate_scores(patches, num_outputs=2, model=model)
    scores[:,-1] *= 20.*spacing_ratio # normalize size in mm
    
    return scores, features


def estimate_lung_position(coordinates, lung_seg_file):
    mask, origin, spacing, orientation = parse_itk_image(load_itk_image(lung_seg_file))
    mask = (mask>0)
     
    # dilate to avoid excluding nodules at the lung boundary
    se_filter = morphology.ball(3, dtype=np.bool8)
    mask=ndimage.morphology.binary_dilation(mask, se_filter)
        
    lung_pixel_count = mask.sum()   
        
    lung_positions = []
    
    order_i = 0
    order_j = 0
    order_k = 0 # convention: 0: bottom->up, 1: top->down        
    if (orientation[0]<0):
        print('x-orientation is %d, inverting order'%(orientation[0]))
        order_i = 1
    if (orientation[4]<0):
        print('x-orientation is %d, inverting order'%(orientation[4]))
        order_j = 1
    if (orientation[8]<0):
        print('z-orientation is %d, inverting order'%(orientation[8]))
        order_k = 1
    
    for i in range(len(coordinates)):
        wc = coordinates[i][0:3]
        kk,jj,ii = worldToVoxelCoord(wc, origin=origin, spacing=spacing, orientation=orientation)
        kk = int(round(kk))
        jj = int(round(jj))
        ii = int(round(ii))
        
        if mask[kk,jj,ii]<=0:
            print('nodule outside lungs:',wc)
        
        if order_i==1:
            ratio_i = mask[:,:,:ii].sum() / lung_pixel_count
        else:
            ratio_i = mask[:,:,ii:].sum() / lung_pixel_count
        if order_j==1:
            ratio_j = mask[:,:jj,:].sum() / lung_pixel_count
        else:
            ratio_j = mask[:,jj:,:].sum() / lung_pixel_count
        if order_k==1:
            ratio_k = mask[:kk,:,:].sum() / lung_pixel_count
        else:
            ratio_k = mask[kk:,:,:].sum() / lung_pixel_count
            
        lung_positions.append([ratio_k,ratio_j,ratio_i,mask[kk,jj,ii]>0])        
    return lung_positions, mask

    
def load_itk_image(filename):
    itkimage = sitk.ReadImage(filename)
    return itkimage

    
def parse_itk_image(itk_image):
    numpyImage = sitk.GetArrayFromImage(itk_image)
    numpyOrigin = np.array(list(itk_image.GetOrigin()))
    numpySpacing = np.array(list(itk_image.GetSpacing()))
    numpyOrientation = np.array(list(itk_image.GetDirection()))
    return numpyImage, numpyOrigin, numpySpacing, numpyOrientation

    
def resample_itk_image(itk_image, target_size, target_spacing, padding_value, interpolator_type=3):
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputDirection(itk_image.GetDirection())
    resampler.SetOutputOrigin(itk_image.GetOrigin())
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(target_size)
    if interpolator_type==3:
        resampler.SetInterpolator(sitk.sitkBSpline)
    elif interpolator_type==1:
        resampler.SetInterpolator(sitk.sitkLinear)
    elif interpolator_type==0:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        print('unknown interpolator',interpolator_type,'choosing b-spline')
        resampler.SetInterpolator(sitk.sitkBSpline)
    resampler.SetDefaultPixelValue(padding_value)
    return resampler.Execute(itk_image)

    
def normalizePlanes(npzarray):
    maxHU = 350.
    minHU = -1150.
    npzarray = (npzarray - minHU) / (maxHU - minHU)
    npzarray[npzarray>1] = 1.
    npzarray[npzarray<0] = 0.
    return npzarray

    
def voxelToWorldCoord(voxel_coord, origin, spacing, orientation):
    world_coord = [
        (voxel_coord[2]*orientation[0]+voxel_coord[1]*orientation[1]+voxel_coord[0]*orientation[2])*spacing[0]+origin[0],
        (voxel_coord[2]*orientation[3]+voxel_coord[1]*orientation[4]+voxel_coord[0]*orientation[5])*spacing[1]+origin[1],
        (voxel_coord[2]*orientation[6]+voxel_coord[1]*orientation[7]+voxel_coord[0]*orientation[8])*spacing[2]+origin[2]
    ]
    return world_coord

    
def worldToVoxelCoord(world_coord, origin, spacing, orientation):
    m = orientation.reshape(3,3)
    m = np.linalg.inv(m)
    m = m.ravel()
    wo = world_coord - origin
    voxel_coord = [
        (m[6]*wo[0]+m[7]*wo[1]+m[8]*wo[2])/spacing[2],
        (m[3]*wo[0]+m[4]*wo[1]+m[5]*wo[2])/spacing[1],
        (m[0]*wo[0]+m[1]*wo[1]+m[2]*wo[2])/spacing[0]
    ]
    return voxel_coord

    
def screen_itk_volume(itk_file, models, do_normalize=True, min_candidates=-1, target_spacing = [0.625,0.625,0.625]):
    itk_image = load_itk_image(itk_file)
    volume_orig, origin, spacing, orientation = parse_itk_image(itk_image)    
    
    if ~(spacing==target_spacing).all():
        # resample using itk
        print('resampling itk volume')
        padding_value = volume_orig.min()
        img_z_orig, img_y_orig, img_x_orig = volume_orig.shape
        img_z_new = int(np.round(img_z_orig*spacing[2]/target_spacing[2]))
        img_y_new = int(np.round(img_y_orig*spacing[1]/target_spacing[1]))
        img_x_new = int(np.round(img_x_orig*spacing[0]/target_spacing[0]))

        itk_image = resample_itk_image(itk_image, [img_x_new,img_y_new,img_z_new], target_spacing, int(padding_value))
        
    volume, _, _, _ = parse_itk_image(itk_image)
    
    if do_normalize:
        print('normalizing volume')
        volume = volume.astype(np.float32)
        # normalize
        volume = normalizePlanes(volume)
    
    candidates, score_map = screen_volume(volume, models, min_candidates=min_candidates)
    
    p = []
    for i in range(len(candidates)):
        p.append(voxelToWorldCoord(candidates[i], origin, target_spacing, orientation))
    
    return p, candidates, score_map


# input is expected in float, normalized between 0 and 1, voxel size = 0.625**3
def screen_volume(volume, models, min_candidates=-1):
    img_z, img_y, img_x = volume.shape
    
    # pad to next 48 multiple
    padz = int(PATCH_SIZE*math.ceil(float(img_z)/PATCH_SIZE))-img_z
    pady = int(PATCH_SIZE*math.ceil(float(img_y)/PATCH_SIZE))-img_y
    padx = int(PATCH_SIZE*math.ceil(float(img_x)/PATCH_SIZE))-img_x
    volume_padded = np.pad(volume, ((0, padz), (0, pady), (0, padx)), 
                                    'constant', constant_values=((0, 0),(0, 0),(0, 0)))

    # compute score map
    score_map = np.zeros(shape=volume_padded.shape, dtype=np.float32)
    for i in range(len(models)):
        score_map += scan_volume(volume=volume_padded, model=models[i], with_bn=True)
    
    # average score map
    score_map /= len(models)
    
    # un-pad
    if (padz>0):
        score_map = score_map[:-padz,:,:]
    if (pady>0):
        score_map = score_map[:,:-pady,:]
    if (padx>0):
        score_map = score_map[:,:,:-padx]

    threshold = 0.5
    
    score_mask = score_map>threshold
    
    # remove smaller blobs using morphological opening
    se_filter = morphology.ball(1, dtype=np.bool8)
    score_mask = ndimage.morphology.binary_opening(score_mask, structure=se_filter)
    
    # find local minima
    coordinates = peak_local_max(score_map*score_mask, min_distance=8)
    
    step = 0.05
    eps = 0.01
    if (min_candidates>-1 and len(coordinates)<min_candidates):
        while (len(coordinates)<min_candidates and threshold>=eps):
            threshold -= step
            if threshold < step:
                print('threshold too low, breaking')
                break
            print('reducing threshold to',threshold,'(number of candidates: %d)'%len(coordinates))
            
            score_mask = score_map>threshold
    
            # remove smaller blobs using morphological opening
            se_filter = morphology.ball(1, dtype=np.bool8)
            score_mask = ndimage.morphology.binary_opening(score_mask, structure=se_filter)
    
            # find local minima
            coordinates = peak_local_max(score_map*score_mask, min_distance=8)
    
    # refine scoremap aroud local minima
    score_map, candidates = refine_scoremap(volume, coordinates, 8, models, max_ite=5, with_bn=True) 
    
    return candidates, score_map
    

# input is expected in float, normalized between 0 and 1, voxel size = 0.625**3
def screen_volume_emphyseme(volume, models, min_candidates=-1, map_index=2):
    img_z, img_y, img_x = volume.shape
    
    patch_size = 64
    
    # pad to next 64 multiple
    padz = int(patch_size*math.ceil(float(img_z)/patch_size))-img_z
    pady = int(patch_size*math.ceil(float(img_y)/patch_size))-img_y
    padx = int(patch_size*math.ceil(float(img_x)/patch_size))-img_x
    volume_padded = np.pad(volume, ((0, padz), (0, pady), (0, padx)), 
                                    'constant', constant_values=((0, 0),(0, 0),(0, 0)))

    # compute score map
    score_map = np.zeros(shape=volume_padded.shape, dtype=np.float32)
    for i in range(len(models)):
        maps = scan_volume_emphyseme(volume=volume_padded, model=models[i])
        score_map += maps[map_index] # consider only severe emphyseme
    
    # average score map
    score_map /= len(models)
    
    # un-pad
    if (padz>0):
        score_map = score_map[:-padz,:,:]
    if (pady>0):
        score_map = score_map[:,:-pady,:]
    if (padx>0):
        score_map = score_map[:,:,:-padx]

    threshold = 0.5
    
    score_mask = score_map>threshold
    
    # remove smaller blobs using morphological opening
    se_filter = morphology.ball(1, dtype=np.bool8)
    score_mask = ndimage.morphology.binary_opening(score_mask, structure=se_filter)
    
    # find local minima
    candidates = peak_local_max(score_map*score_mask, min_distance=8)
    
    step = 0.05
    eps = 0.01
    if (min_candidates>-1 and len(candidates)<min_candidates):
        while (len(candidates)<min_candidates and threshold>=eps):
            threshold -= step
            if threshold < step:
                print('threshold too low, breaking')
                break
            print('reducing threshold to',threshold,'(number of candidates: %d)'%len(candidates))
            
            score_mask = score_map>threshold
    
            # remove smaller blobs using morphological opening
            se_filter = morphology.ball(1, dtype=np.bool8)
            score_mask = ndimage.morphology.binary_opening(score_mask, structure=se_filter)
    
            # find local minima
            candidates = peak_local_max(score_map*score_mask, min_distance=8)
    
    tmap = np.zeros(score_mask.shape, dtype=np.int32)
    for i in range(len(candidates)):
        kk,jj,ii = candidates[i]
        tmap[kk,jj,ii] = 1
        
    labels, nb_labels = ndimage.label(tmap)
    candidates = ndimage.measurements.center_of_mass(tmap, labels, np.arange(1,nb_labels+1))
    candidates = np.round(candidates).astype(int)
    
    return candidates, score_map



# input is expected in float, normalized between 0 and 1, voxel size = 0.625**3
def screen_volume_lung_segmentation(volume, models, min_candidates=-1):
    img_z, img_y, img_x = volume.shape
    
    patch_size = 64
    offset=patch_size//2
    
    volume = np.pad(volume, ((offset,offset),(offset,offset),(offset,offset)), # pad to center
                    'constant', constant_values=((0, 0),(0, 0),(0, 0)))  
    
    # pad to next 64 multiple
    padz = int(patch_size*math.ceil(float(img_z)/patch_size))-img_z
    pady = int(patch_size*math.ceil(float(img_y)/patch_size))-img_y
    padx = int(patch_size*math.ceil(float(img_x)/patch_size))-img_x
    volume_padded = np.pad(volume, ((0, padz), (0, pady), (0, padx)), 
                                    'constant', constant_values=((0, 0),(0, 0),(0, 0)))

    # compute score map
    score_map = np.zeros(shape=volume_padded.shape, dtype=np.float32)
    for i in range(len(models)):
        maps = scan_volume_lung_segmentation(volume=volume_padded, model=models[i])
        score_map += maps
    
    # average score map
    score_map /= len(models)
    
    # un-pad
    if (padz>0):
        score_map = score_map[:-padz,:,:]
    if (pady>0):
        score_map = score_map[:,:-pady,:]
    if (padx>0):
        score_map = score_map[:,:,:-padx]

    score_map = score_map[offset:-offset,offset:-offset,offset:-offset]
        
    threshold = 0.5
    
    score_mask = score_map>threshold
    
    # remove smaller blobs using morphological opening
    se_filter = morphology.ball(1, dtype=np.bool8)
    score_mask = ndimage.morphology.binary_opening(score_mask, structure=se_filter)
    
    # find local minima
    candidates = peak_local_max(score_map*score_mask, min_distance=8)
    
    step = 0.05
    eps = 0.01
    if (min_candidates>-1 and len(candidates)<min_candidates):
        while (len(candidates)<min_candidates and threshold>=eps):
            threshold -= step
            if threshold < step:
                print('threshold too low, breaking')
                break
            print('reducing threshold to',threshold,'(number of candidates: %d)'%len(candidates))
            
            score_mask = score_map>threshold
    
            # remove smaller blobs using morphological opening
            se_filter = morphology.ball(1, dtype=np.bool8)
            score_mask = ndimage.morphology.binary_opening(score_mask, structure=se_filter)
    
            # find local minima
            candidates = peak_local_max(score_map*score_mask, min_distance=8)
    
    tmap = np.zeros(score_mask.shape, dtype=np.int32)
    for i in range(len(candidates)):
        kk,jj,ii = candidates[i]
        tmap[kk,jj,ii] = 1
        
    labels, nb_labels = ndimage.label(tmap)
    candidates = ndimage.measurements.center_of_mass(tmap, labels, np.arange(1,nb_labels+1))
    candidates = np.round(candidates).astype(int)
    
    return candidates, score_map

