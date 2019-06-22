import tensorflow as tf
import layers
import time
import sys

global_seed = 5

class Scorer:
    def __init__(self):
        return
    
    def __call__(self, inp, training, pad=True): # pad by 12 pixels in each side to get 1024x1024 image if the image size is 1000x1000
        a = time.time()
        
        with tf.variable_scope("Scorer"): # define variable scope
            if pad:
                inp = layers.padding_layer(inp, padding=(12, 12), pad_values=0) # 1024x1024
            
            max_pool1 = layers.max_pool_layer(inp, pool_size=(2,2), strides=(2,2)) # 512x512
            max_pool2 = layers.max_pool_layer(max_pool1, pool_size=(2,2), strides=(2,2)) # 256x256
            
            conv1 = layers.conv_block_scorer(max_pool2, training, out_channels=32, filter_size=(4, 4), strides=(2, 2), padding=(1, 1), pad_values=0, use_bias=True, alpha=0.2) # 128x128
            conv2 = layers.conv_block_scorer(conv1, training, out_channels=64, filter_size=(4, 4), strides=(2, 2), padding=(1, 1), pad_values=0, use_bias=True, alpha=0.2) # 64x64
            conv3 = layers.conv_block_scorer(conv2, training, out_channels=128, filter_size=(4, 4), strides=(2, 2), padding=(1, 1), pad_values=0, use_bias=True, alpha=0.2) # 32x32
            conv4 = layers.conv_block_scorer(conv3, training, out_channels=256, filter_size=(4, 4), strides=(2, 2), padding=(1, 1), pad_values=0, use_bias=True, alpha=0.2) # 16x16
            conv5 = layers.conv_block_scorer(conv4, training, out_channels=512, filter_size=(4, 4), strides=(2, 2), padding=(1, 1), pad_values=0, use_bias=True, alpha=0.2) # 8x8
            conv6 = layers.conv_block_scorer(conv5, training, out_channels=1024, filter_size=(4, 4), strides=(2, 2), padding=(1, 1), pad_values=0, use_bias=True, alpha=0.2) # 4x4
            
            flat = tf.reshape(conv6, [-1, 1024*4*4])
            dense1 = layers.dense_block_scorer(flat, training, units=128, use_bias=True, dropout_rate=0.3)
            dense2 = layers.dense_layer(dense1, units=1, use_bias=True)
            
            output = dense2
            
        print("Scorer Model built in {} s".format(time.time()-a))
        return output
        
    def compute_loss(self, scores_gt, scores_pred):
        with tf.name_scope("loss"):
            loss = tf.losses.absolute_difference(labels=scores_gt, predictions=scores_pred, reduction=tf.losses.Reduction.MEAN)
            return loss
    
    def train_op(self, loss, learning_rate, beta1=0.9, beta2=0.999):
        with tf.name_scope("train_op"):
            global_step = tf.Variable(0, name='global_step',trainable=False)
            
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="Scorer")
            with tf.control_dependencies(update_ops): # for batch_norm
                train_op = tf.train.AdamOptimizer(learning_rate, beta1, beta2).minimize(loss, global_step)
        
        return train_op, global_step









