import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

global_seed=5
initializer = xavier_initializer(seed=global_seed)

""" all functions below expect input in channels first format for perfomance on GPU 
    all initializations done with initializer above
"""

def padding_layer(inp, padding, pad_values=-1):
    with tf.name_scope("padding_layer"):
        pad_H = padding[0]
        pad_W = padding[1]
        if pad_H > 0 or pad_W > 0: # perform padding only if padding by 1 pixel or more
            paddings = [[0, 0], [0, 0], [pad_H, pad_H], [pad_W, pad_W]] # pad only along the height and width dimension, no padding along the batch and channels dimensions
            return tf.pad(inp, paddings, constant_values=pad_values)
        else:
            return inp

def conv_layer(inp, out_channels, filter_size=1, strides=1, padding="same", pad_values=-1, use_bias=False, activation=None): 
    with tf.name_scope("conv_layer"):
        if isinstance(padding, (list, tuple)): # if a list/tuple passed
            inp = padding_layer(inp, padding, pad_values)
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
    
def max_pool_layer(inp, pool_size=1, strides=1, padding="valid", pad_values=-1):
    with tf.name_scope("max_pool_layer"):
        if isinstance(padding, (list, tuple)): # if a list/tuple passed
            inp = padding_layer(inp, padding, pad_values)
            padding = "valid"
            
        return tf.layers.max_pooling2d(inp, pool_size=pool_size, strides=strides, padding=padding, data_format="channels_first")

def avg_pool_layer(inp, pool_size=1, strides=1, padding="valid", pad_values=-1):
    with tf.name_scope("avg_pool_layer"):
        if isinstance(padding, (list, tuple)): # if a list/tuple passed
            inp = padding_layer(inp, padding, pad_values)
            padding = "valid"
            
        return tf.layers.average_pooling2d(inp, pool_size=pool_size, strides=strides, padding=padding, data_format="channels_first")

def dropout_layer(inp, training, dropout_rate):
    with tf.name_scope("dropout_layer"):
        return tf.layers.dropout(inp, rate=dropout_rate, seed=global_seed, training=training)

def resize_layer(inp, new_size, resize_method):
    with tf.name_scope("resize_layer"):
        inp = tf.transpose(inp, [0, 2, 3, 1]) # convert to channels_last format
        
        out =  tf.image.resize_images(inp, size=new_size, method=resize_method)

        return tf.transpose(out, [0, 3, 1, 2]) # convert to channels_first format and return result
        
# ------------------------------------------------ Blocks -------------------------------------------------
# CGAN
def deconv_block_cgan(inp, training, momentum, out_channels, filter_size=1, strides=1, padding="same", use_bias=False):
    with tf.name_scope("deconv_block"):
        tmp = batch_norm_layer(inp, training, momentum)
        tmp = leaky_relu_layer(tmp)
        out = deconv_layer(tmp, out_channels, filter_size, strides, padding, use_bias)
    return out

def conv_block_cgan(inp, training, dropout_rate, out_channels, filter_size=1, strides=1, padding="same", use_bias=False):
    with tf.name_scope("conv_block"):
        tmp = conv_layer(inp, out_channels, filter_size, strides, padding, use_bias)
        tmp = leaky_relu_layer(tmp)
        out = dropout_layer(tmp, training, dropout_rate)
    return out

def dense_block_cgan(inp, training, units, dropout_rate, use_bias=False):
    with tf.name_scope("dense_block"):
        return dropout_layer(leaky_relu_layer(dense_layer(inp, units, use_bias)), training, dropout_rate)

# DCGAN
def deconv_block_dcgan(inp, training, out_channels, filter_size=1, strides=1, padding="same", use_bias=False):
    with tf.name_scope("deconv_block"):
        tmp = deconv_layer(inp, out_channels, filter_size, strides, padding, use_bias)
        tmp = batch_norm_layer(tmp, training)
        out = relu_layer(tmp)
    return out
    
def conv_block_dcgan(inp, training, out_channels, filter_size=1, strides=1, padding="same", pad_values=-1, use_bias=False, alpha=0.2):
    with tf.name_scope("conv_block"):
        tmp = conv_layer(inp, out_channels, filter_size, strides, padding, pad_values, use_bias)
        tmp = batch_norm_layer(tmp, training)
        out = leaky_relu_layer(tmp, alpha)
#        out = dropout_layer(out, training, 0.3)
    return out

# FullresGAN
def deconv_block_fullres(inp, training, out_channels, filter_size=1, strides=1, padding="same", use_bias=False):
    with tf.name_scope("deconv_block"):
        tmp = deconv_layer(inp, out_channels, filter_size, strides, padding, use_bias)
        tmp = batch_norm_layer(tmp, training)
        out = leaky_relu_layer(tmp)
    return out
    
def conv_block_fullres(inp, training, out_channels, filter_size=1, strides=1, padding="same", pad_values=-1, use_bias=False, alpha=0.3, dropout_rate=0.5):
    with tf.name_scope("conv_block"):
        tmp = conv_layer(inp, out_channels, filter_size, strides, padding, pad_values, use_bias)
        tmp = leaky_relu_layer(tmp, alpha)
        out = dropout_layer(tmp, training, dropout_rate)
    return out

def dense_block_fullres(inp, training, units, use_bias=False, alpha=0.3, dropout_rate=0.5):
    with tf.name_scope("dense_block"):
        tmp = dense_layer(inp, units, use_bias)
        tmp = leaky_relu_layer(tmp, alpha)
        out = dropout_layer(tmp, training, dropout_rate)   
    return out

#SRM
def residual_block_srm(inp, training, out_channels, pad_values, use_bias):
    with tf.name_scope("residual_block_srm"):
        tmp = conv_layer(inp, out_channels, filter_size=(3, 3), strides=(1, 1), padding=(1, 1), pad_values=pad_values, use_bias=use_bias)
        tmp = batch_norm_layer(tmp, training)
        tmp = relu_layer(tmp)

        tmp = conv_layer(tmp, out_channels, filter_size=(3, 3), strides=(1, 1), padding=(1, 1), pad_values=pad_values, use_bias=use_bias)
        tmp = batch_norm_layer(tmp, training)
        tmp = tmp + inp # residual connection
        tmp = relu_layer(tmp)
        return tmp
    
def residual_module_srm(inp, training, out_channels, nb_blocks, pad_values=0, use_bias=False):
    with tf.name_scope("residual_module_srm"):
        tmp = inp
        
        for i in range(nb_blocks):
            tmp = residual_block_srm(tmp, training, out_channels, pad_values=pad_values, use_bias=use_bias)
        
        tmp = conv_layer(tmp, out_channels, filter_size=(3, 3), strides=(1, 1), padding=(1, 1), pad_values=pad_values, use_bias=use_bias)
        tmp = batch_norm_layer(tmp, training)
        tmp = tmp + inp # residual connection
        tmp = relu_layer(tmp)
        return tmp
        
# Scorer
def conv_block_scorer(inp, training, out_channels, filter_size=1, strides=1, padding="same", pad_values=0, use_bias=False, alpha=0.2):
    with tf.name_scope("conv_block"):
        tmp = conv_layer(inp, out_channels, filter_size, strides, padding, pad_values, use_bias)
        tmp = batch_norm_layer(tmp, training)
        out = leaky_relu_layer(tmp, alpha)
    return out
    
def dense_block_scorer(inp, training, units, use_bias=False, dropout_rate=0.0):
    with tf.name_scope("dense_block"):
        tmp = dense_layer(inp, units, use_bias)
        tmp = relu_layer(tmp)
        out = dropout_layer(tmp, training, dropout_rate)   
    return out
        
     
# MCGAN
def deconv_block_mcgan(inp, training, momentum, out_channels, filter_size=1, strides=1, padding="same", use_bias=False, batch_norm=True):
    with tf.name_scope("deconv_block"):
        tmp = deconv_layer(inp, out_channels, filter_size, strides, padding, use_bias)
        if batch_norm:
            tmp = batch_norm_layer(tmp, training, momentum)
        out = relu_layer(tmp)
    return out

def conv_block_mcgan(inp, training, momentum, out_channels, filter_size=1, strides=1, padding="same", pad_value=-1, use_bias=False, batch_norm=True, alpha=0.3):
    with tf.name_scope("conv_block"):
        tmp = conv_layer(inp, out_channels, filter_size, strides, padding, pad_value, use_bias)
        if batch_norm:
            tmp = batch_norm_layer(tmp, training, momentum)
        out = leaky_relu_layer(tmp, alpha)
        #out = dropout_layer(tmp, training, dropout_rate)
    return out

def dense_block_mcgan(inp, training, units, dropout_rate, use_bias=False):
    with tf.name_scope("dense_block"):
        return dropout_layer(leaky_relu_layer(dense_layer(inp, units, use_bias)), training, dropout_rate)

def batch_norm_layer_mcgan(inp, training, momentum=0.99, epsilon=0.001):
    with tf.name_scope("batch_norm_layer"):
        return tf.layers.batch_normalization(inputs=inp, axis=1, momentum=momentum, epsilon=epsilon,
                                             training=training)

def minibatch(inp, num_kernels=5, kernel_dim=3):
    with tf.name_scope("minibatch"):
        x = dense_layer(inp, num_kernels * kernel_dim, True)
        activation = tf.reshape(x, (-1, num_kernels, kernel_dim))
        diffs = tf.expand_dims(activation, 3) - \
            tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
        abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)
        minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
        return tf.concat([inp, minibatch_features], axis=1)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



