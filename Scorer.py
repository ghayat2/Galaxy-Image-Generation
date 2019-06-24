import tensorflow as tf
import layers
import time
import sys

global_seed = 5

class Scorer:
    def __init__(self):
        return
    
    def __call__(self, inp, training, pad=True, zero_centered=False): # pad by 12 pixels in each side to get 1024x1024 image if the image size is 1000x1000
        a = time.time()
        pad_value = 0
        if zero_centered:
            pad_value = -1
        
        with tf.variable_scope("Scorer"): # define variable scope
        
#            histograms = tf.map_fn(lambda a: tf.cast(tf.histogram_fixed_width((a+1)*128.0 if zero_centered else (a*255.0), value_range=[0.0, 255.0], nbins=10), tf.float32), inp)

            if pad:
                inp = layers.padding_layer(inp, padding=(12, 12), pad_values=pad_value) # 1024x1024
            
            max_pool1 = layers.max_pool_layer(inp, pool_size=(2,2), strides=(2,2)) # 512x512
            max_pool2 = layers.max_pool_layer(max_pool1, pool_size=(2,2), strides=(2,2)) # 256x256
            max_pool3 = layers.max_pool_layer(max_pool2, pool_size=(2,2), strides=(2,2)) # 128x128
            max_pool4 = layers.max_pool_layer(max_pool3, pool_size=(2,2), strides=(2,2)) # 64x64
            
            resized = layers.resize_layer(inp, new_size=[64, 64], resize_method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            
            concat1 = tf.concat([max_pool4, resized], axis=1)

            
#            conv1 = layers.conv_block_scorer(inp, training, out_channels=8, filter_size=(4, 4), strides=(2, 2), padding=(1, 1), pad_values=pad_value, use_bias=True, alpha=0.2) # 512x512
#            conv2 = layers.conv_block_scorer(conv1, training, out_channels=16, filter_size=(4, 4), strides=(2, 2), padding=(1, 1), pad_values=pad_value, use_bias=True, alpha=0.2) # 256x256
#            conv3 = layers.conv_block_scorer(conv2, training, out_channels=32, filter_size=(4, 4), strides=(2, 2), padding=(1, 1), pad_values=pad_value, use_bias=True, alpha=0.2) # 128x128
#            conv4 = layers.conv_block_scorer(conv3, training, out_channels=64, filter_size=(4, 4), strides=(2, 2), padding=(1, 1), pad_values=pad_value, use_bias=True, alpha=0.2) # 64x64
#            
#            concat2 = tf.concat([concat1, conv4], axis=1)
            
            conv5 = layers.conv_block_scorer(concat1, training, out_channels=128, filter_size=(4, 4), strides=(2, 2), padding=(1, 1), pad_values=pad_value, use_bias=True, alpha=0.2) # 32x32
            conv6 = layers.conv_block_scorer(conv5, training, out_channels=256, filter_size=(4, 4), strides=(2, 2), padding=(1, 1), pad_values=pad_value, use_bias=True, alpha=0.2) # 16x16
            conv7 = layers.conv_block_scorer(conv6, training, out_channels=512, filter_size=(4, 4), strides=(2, 2), padding=(1, 1), pad_values=pad_value, use_bias=True, alpha=0.2) # 8x8
            conv8 = layers.conv_block_scorer(conv7, training, out_channels=1024, filter_size=(4, 4), strides=(2, 2), padding=(1, 1), pad_values=pad_value, use_bias=True, alpha=0.2) # 4x4

            flat = tf.reshape(conv8, [-1, 1024*4*4])
#            concat3 = tf.concat([flat, histograms], axis=-1)
            
            dense1 = layers.dense_block_scorer(flat, training, units=1024, use_bias=True, dropout_rate=0.3)
#            dense2 = layers.dense_block_scorer(dense1, training, units=256, use_bias=True, dropout_rate=0.3)
#            dense3 = layers.dense_block_scorer(dense2, training, units=128, use_bias=True, dropout_rate=0.3)
            dense4 = layers.dense_layer(dense1, units=1, use_bias=True)
            
#            print(dense4.shape)
#            sys.exit(0)
            output = dense4
            
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









