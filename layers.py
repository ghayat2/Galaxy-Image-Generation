import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

global_seed=5
initializer = xavier_initializer(seed=global_seed)

""" all functions below expect input in channels first format for perfomance on GPU 
    all initializations done with initializer above
"""

def padding_layer(inp, padding):
    with tf.name_scope("padding_layer"):
        pad_H = padding[0]
        pad_W = padding[1]
        if pad_H > 0 or pad_W > 0: # perform padding only if padding by 1 pixel or more
            paddings = [[0, 0], [0, 0], [pad_H, pad_H], [pad_W, pad_W]] # pad only along the height and width dimension, no padding along the batch and channels dimensions
            return tf.pad(inp, paddings)
        else:
            return inp

def conv_layer(inp, out_channels, filter_size=1, strides=1, padding="same", use_bias=False, activation=None): 
    with tf.name_scope("conv_layer"):
        if isinstance(padding, (list, tuple)): # if a list/tuple passed
            inp = padding_layer(inp, padding)
            padding = "valid"

        return tf.layers.conv2d(inputs=inp, filters=out_channels, kernel_size=filter_size, strides=strides, padding=padding, activation=activation,
                                data_format="channels_first", use_bias=use_bias, kernel_initializer=initializer, 
                                bias_initializer=initializer)
                                
def deconv_layer(inp, out_channels, filter_size=1, strides=1, padding="same", use_bias=False, activation=None): 
    with tf.name_scope("deconv_layer"):
        pad = padding
        if isinstance(padding, (list, tuple)): # if a list/tuple passed
            pad = "valid"
            
        out = tf.layers.conv2d_transpose(inputs=inp, filters=out_channels, kernel_size=filter_size, strides=strides, padding=pad, activation=activation,
                                data_format="channels_first", use_bias=use_bias, kernel_initializer=initializer, 
                                bias_initializer=initializer)
                                
        if isinstance(padding, (list, tuple)): # if a list/tuple passed, remove padding since deconv
            pad_H = padding[0]
            pad_W = padding[1]
            out = out[:, :, pad_H:-pad_H, pad_W:-pad_W]
        
        return out
            
def relu_layer(inp):
    with tf.name_scope("relu_layer"):
        return tf.nn.relu(inp)
        
def tanh_layer(inp):
    with tf.name_scope("tanh_layer"):
        return tf.nn.tanh(inp)
        
def leaky_relu_layer(inp, alpha=0.3):
    with tf.name_scope("leaky_relu_layer"):
        return tf.nn.leaky_relu(inp, alpha=alpha)

def dense_layer(inp, units, use_bias=False):
    with tf.name_scope("dense_layer"):
        return tf.layers.dense(inp, units, kernel_initializer=initializer, bias_initializer=initializer)

def batch_norm_layer(inp, training, momentum=0.99, epsilon=0.001):
    with tf.name_scope("batch_norm_layer"):
        return tf.layers.batch_normalization(inputs=inp, axis=1, momentum=momentum, epsilon=epsilon, training=training) # axis=1 because channels first
    
def max_pool_layer(inp, pool_size=1, strides=1, padding="valid"):
    with tf.name_scope("max_pool_layer"):
        if isinstance(padding, (list, tuple)): # if a list/tuple passed
            inp = padding_layer(inp, padding)
            padding = "valid"
            
        return tf.layers.max_pooling2d(inp, pool_size=pool_size, strides=strides, padding=padding, data_format="channels_first")

def dropout_layer(inp, training, dropout_rate):
    with tf.name_scope("dropout_layer"):
        return tf.layers.dropout(inp, rate=dropout_rate, seed=global_seed, training=training)

def nearest_neighbor_down_sampling_layer(inp, new_size):
    with tf.name_scope("nearest_neighbor_down_sampling_layer"):
        inp = tf.transpose(inp, [0, 2, 3, 1]) # convert to channels_last format
        
        out =  tf.image.resize_nearest_neighbor(inp, new_size) # resize images with nearest neighbor up sampling

        return tf.transpose(out, [0, 3, 1, 2]) # convert to channels_first format and return result
        
# ------------------------------------------------ Blocks -------------------------------------------------
def deconv_block(inp, training, momentum, out_channels, filter_size=1, strides=1, padding="same", use_bias=False):
    with tf.name_scope("deconv_block"):
        tmp = deconv_layer(inp, out_channels, filter_size, strides, padding, use_bias)
        tmp = leaky_relu_layer(tmp)
        out = batch_norm_layer(tmp, training, momentum)
    return out

def conv_block(inp, training, dropout_rate, out_channels, filter_size=1, strides=1, padding="same", use_bias=False):
    with tf.name_scope("conv_block"):
        tmp = conv_layer(inp, out_channels, filter_size, strides, padding, use_bias)
        tmp = leaky_relu_layer(tmp)
        out = dropout_layer(tmp, training, dropout_rate)
    return out

def dense_block(inp, training, units, dropout_rate, use_bias=False):
    with tf.name_scope("dense_block"):
        return dropout_layer(leaky_relu_layer(dense_layer(inp, units, use_bias)), training, dropout_rate)

def deconv_block_dcgan(inp, training, out_channels, filter_size=1, strides=1, padding="same", use_bias=False):
    with tf.name_scope("deconv_block"):
        tmp = deconv_layer(inp, out_channels, filter_size, strides, padding, use_bias)
        tmp = batch_norm_layer(tmp, training)
        out = relu_layer(tmp)
    return out
    
def conv_block_dcgan(inp, training, out_channels, filter_size=1, strides=1, padding="same", use_bias=False, alpha=0.2):
    with tf.name_scope("conv_block"):
        tmp = conv_layer(inp, out_channels, filter_size, strides, padding, use_bias)
        tmp = batch_norm_layer(tmp, training)
        out = leaky_relu_layer(tmp)
#        out = dropout_layer(out, training, 0.3)
    return out




