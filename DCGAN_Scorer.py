import tensorflow as tf
import layers
import time

global_seed = 5

class Scorer_head:
    def __init__(self):
        return
        
    def scorer_head_model(self, features, training, reuse=False):
        with tf.variable_scope("scorer_head", reuse=reuse): # define variable scope
#            dense1 = layers.dense_layer(features, units=1024, use_bias=True)
#            relu1 = layers.relu_layer(dense1)
#            dense2 = layers.dense_layer(features, units=512, use_bias=True)
#            relu2 = layers.relu_layer(dense2)
            dense3 = layers.dense_layer(features, units=256, use_bias=True)
            relu3 = layers.relu_layer(dense3)
            dense4 = layers.dense_layer(relu3, units=1, use_bias=True)
            
            
        return dense4
        
    def compute_loss(self, scores_gt, scores_pred):
        with tf.name_scope("loss"):
            loss = tf.losses.absolute_difference(labels=scores_gt, predictions=scores_pred, reduction=tf.losses.Reduction.MEAN)
            return loss

    def scorer_head_vars(self):
        head_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="scorer_head")
        return head_vars
        
    def train_op(self, loss, learning_rate, beta1, beta2, var_list, scope):
        with tf.name_scope("train_op"):
            global_step = tf.Variable(0, name='global_step',trainable=False)
            
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope)
            with tf.control_dependencies(update_ops): # for batch_norm
                train_op = tf.train.AdamOptimizer(learning_rate, beta1, beta2).minimize(loss, global_step, var_list)
        
        return train_op, global_step
        
    def get_scope(self):
        return "scorer_head"
