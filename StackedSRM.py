import tensorflow as tf
import layers
import time
import sys

global_seed = 5

class StackedSRM:
    def __init__(self, nb_stacks):
        self.nb_stacks = nb_stacks
    
    def __call__(self, inp, training):
        a = time.time()
        
        with tf.variable_scope("StackedSRM"): # define variable scope
            inter = inp # intermediate input
            outputs = []
            for i in range(self.nb_stacks):
                if i < 3:
                    with tf.name_scope("stack"):
                        conv = layers.conv_layer(inter, out_channels=64, filter_size=(4, 4), strides=(2, 2), padding=(1, 1), pad_values=0, use_bias=False)
                        relu = layers.relu_layer(conv)
                        res_module = layers.residual_module_srm(relu, training, out_channels=64, nb_blocks=6, pad_values=0, use_bias=False)
                        
                        h, w = tf.shape(res_module)[2], tf.shape(res_module)[3]
                        up_sample1 = layers.resize_layer(res_module, new_size=[2*h, 2*w], resize_method=tf.image.ResizeMethod.NEAREST_NEIGHBOR) # nearest neighbor up sampling
                        conv1 = layers.conv_layer(up_sample1, out_channels=128, filter_size=(3, 3), strides=(1, 1), padding=(1, 1), pad_values=0, use_bias=False)
                        
                        h, w = tf.shape(conv1)[2], tf.shape(conv1)[3]
                        up_sample2 = layers.resize_layer(conv1, new_size=[2*h, 2*w], resize_method=tf.image.ResizeMethod.NEAREST_NEIGHBOR) # nearest neighbor up sampling
                        conv2 = layers.conv_layer(up_sample2, out_channels=1, filter_size=(3, 3), strides=(1, 1), padding=(1, 1), pad_values=0, use_bias=False)
                        
                        inter = (layers.tanh_layer(conv2)+1)/2.0 # apply tanh and renormalize so that the output is in the range [0, 1] to prepare it to be inputted to the next stack
                else: # the fourth and last stack
                    h, w = tf.shape(inter)[2], tf.shape(inter)[3]
                    
                    conv3 = layers.conv_layer(inter, out_channels=16, filter_size=(3, 3), strides=(1, 1), padding=(1, 1), pad_values=0, use_bias=False)
                    up_sample3 = layers.resize_layer(conv3, new_size=[2*h, 2*w], resize_method=tf.image.ResizeMethod.NEAREST_NEIGHBOR) # nearest neighbor up sampling
                    
                    
                    conv4 = layers.conv_layer(up_sample3, out_channels=1, filter_size=(3, 3), strides=(1, 1), padding=(1, 1), pad_values=0, use_bias=False)
                    
                    inter = layers.tanh_layer(conv4) # apply tanh and renormalize so that the output is in the range [0, 1]
                    
                outputs.append(inter)
        
        print("SRM Model built in {} s".format(time.time()-a))
        return outputs
        
    def compute_loss(self, outputs_gt, outputs_pred):
        with tf.name_scope("loss"):
            loss = 0
            for i in range(self.nb_stacks):
                loss += tf.losses.mean_squared_error(outputs_gt[i], outputs_pred[i])
            return loss
    
    def train_op(self, loss, learning_rate, beta1, beta2):
        with tf.name_scope("train_op"):
            global_step = tf.Variable(0, name='global_step',trainable=False)
            
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="StackedSRM")
            with tf.control_dependencies(update_ops): # for batch_norm
                train_op = tf.train.AdamOptimizer(learning_rate, beta1, beta2).minimize(loss, global_step)
        
        return train_op, global_step









