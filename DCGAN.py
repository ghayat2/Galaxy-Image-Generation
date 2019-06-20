import tensorflow as tf
import layers
import time

global_seed = 5

class DCGAN:
    def __init__(self):
        return
    
    def generator_model(self, noise, training, reuse=False): # construct the graph of the generator
        a = time.time()
        
        with tf.variable_scope("generator", reuse=reuse): # define variable scope to easily retrieve vars of the generator
            
            gen_inp = noise
            with tf.name_scope("preprocess_inp"):
                dense1 = layers.dense_layer(gen_inp, units=4*4*1024, use_bias=True)
                relu1 = layers.relu_layer(dense1)
                reshaped = tf.reshape(relu1, shape=[-1, 1024, 4, 4]) # shape=(batch_size, 1024, 4, 4)
                
            deconv1 = layers.deconv_block_dcgan(reshaped, training, out_channels=512, filter_size=(4,4), strides=(2,2), padding="same", use_bias=True) # shape=(batch_size, 512, 8, 8)
            deconv2 = layers.deconv_block_dcgan(deconv1, training, out_channels=256, filter_size=(4,4), strides=(2,2), padding="same", use_bias=True) # shape=(batch_size, 256, 16, 16)
            deconv3 = layers.deconv_block_dcgan(deconv2, training, out_channels=128, filter_size=(4,4), strides=(2,2), padding="same", use_bias=True) # shape=(batch_size, 128, 32, 32)
            deconv4 = layers.deconv_block_dcgan(deconv3, training, out_channels=64, filter_size=(4,4), strides=(2,2), padding="same", use_bias=True) # shape=(batch_size, 64, 64, 64)
            deconv5 = layers.deconv_block_dcgan(deconv4, training, out_channels=32, filter_size=(4,4), strides=(2,2), padding="same", use_bias=True) # shape=(batch_size, 32, 128, 128)
            deconv6 = layers.deconv_block_dcgan(deconv5, training, out_channels=16, filter_size=(4,4), strides=(2,2), padding="same", use_bias=True) # shape=(batch_size, 16, 256, 256)
            deconv7 = layers.deconv_block_dcgan(deconv6, training, out_channels=8, filter_size=(4,4), strides=(2,2), padding="same", use_bias=True) # shape=(batch_size, 8, 512, 512)
            deconv8 = layers.deconv_layer(deconv7, out_channels=1, filter_size=(4,4), strides=(2,2), padding=(13, 13), use_bias=True) # shape=(batch_size, 1, 64, 64)
                
            gen_out = layers.tanh_layer(deconv8)
        print("Built Generator model in {} s".format(time.time()-a))
        list_ops = [dense1, relu1, reshaped, deconv1, deconv2, deconv3, deconv4, deconv5, deconv6, deconv7, deconv8, gen_out] # list of operations, can be used to run the graph up to a certain op
                                                                                                                     # i,e get the subgraph
        return gen_out, list_ops
        
    def discriminator_model(self, inp, training, reuse=False, resize=False): # construct the graph of the discriminator
        a = time.time()
        with tf.variable_scope("discriminator", reuse=reuse): # define variable scope to easily retrieve vars of the discriminator
            
#            if resize:
#                inp_scaled = layers.nearest_neighbor_down_sampling_layer(inp, new_size=[64, 64]) # shape=(batch_size, 1, 64, 64)
#            else:
#                inp_scaled = inp
            conv1 = layers.conv_block_dcgan(inp, training, out_channels=8, filter_size=(4, 4), strides=(2, 2), padding=(13, 13), use_bias=True, alpha=0.2) # shape=(batch_size, 8, 512, 512)
            conv2 = layers.conv_block_dcgan(conv1, training, out_channels=16, filter_size=(4, 4), strides=(2, 2), padding="same", use_bias=True, alpha=0.2) # shape=(batch_size, 16, 256, 256)
            conv3 = layers.conv_block_dcgan(conv2, training, out_channels=32, filter_size=(4, 4), strides=(2, 2), padding="same", use_bias=True, alpha=0.2) # shape=(batch_size, 32, 128, 128)
            conv4 = layers.conv_block_dcgan(conv3, training, out_channels=64, filter_size=(4, 4), strides=(2, 2), padding="same", use_bias=True, alpha=0.2) # shape=(batch_size, 64, 64, 64)
            conv5 = layers.conv_block_dcgan(conv4, training, out_channels=128, filter_size=(4, 4), strides=(2, 2), padding="same", use_bias=True, alpha=0.2) # shape=(batch_size, 128, 32, 32)
            conv6 = layers.conv_block_dcgan(conv5, training, out_channels=256, filter_size=(4, 4), strides=(2, 2), padding="same", use_bias=True, alpha=0.2) # shape=(batch_size, 256, 16, 16)
            conv7 = layers.conv_block_dcgan(conv6, training, out_channels=512, filter_size=(4, 4), strides=(2, 2), padding="same", use_bias=True, alpha=0.2) # shape=(batch_size, 512, 8, 8)
            conv8 = layers.conv_block_dcgan(conv7, training, out_channels=1024, filter_size=(4, 4), strides=(2, 2), padding="same", use_bias=True, alpha=0.2) # shape=(batch_size, 1024, 4, 4)
            flat = tf.reshape(conv8, [-1, 1024*4*4])
            logits = layers.dense_layer(flat, units=2, use_bias=True)
            
            out = logits
        print("Built Discriminator model in {} s".format(time.time()-a))
        list_ops = [conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, flat, logits, out]
        
        return out, list_ops
        
    def generator_vars(self):
        gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")
        return gen_vars

    def discriminator_vars(self):
        discr_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")
        return discr_vars

    def generator_loss(self, fake_out, labels):
        with tf.name_scope("generator_loss"):
            loss_gen = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=fake_out))
        return loss_gen
        
    def discriminator_loss(self, fake_out, real_out, fake_labels, real_labels):
        with tf.name_scope("discriminator_loss"):
            fake_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=fake_labels, logits=fake_out)
            real_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real_labels, logits=real_out)
            total_loss = tf.reduce_mean(fake_loss + real_loss)
        return total_loss  

    def train_op(self, loss, learning_rate, beta1, beta2, var_list, scope):
        with tf.name_scope("train_op"):
            global_step = tf.Variable(0, name='global_step',trainable=False)
            
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope)
            with tf.control_dependencies(update_ops): # for batch_norm
                train_op = tf.train.AdamOptimizer(learning_rate, beta1, beta2).minimize(loss, global_step, var_list)
        
        return train_op, global_step


























