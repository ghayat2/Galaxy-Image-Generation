import tensorflow as tf
import layers
import time

global_seed = 5

class FullresGAN:
    def __init__(self):
        return
    
    def generator_model(self, noise, training, reuse=False): # construct the graph of the generator
        a = time.time()
        
        with tf.variable_scope("generator", reuse=reuse): # define variable scope to easily retrieve vars of the generator
            
            gen_inp = noise
            with tf.name_scope("preprocess_inp"):
                dense1 = layers.dense_layer(gen_inp, units=5*5*256, use_bias=True)
                batch1 = layers.batch_norm_layer(dense1, training=training, momentum=0.8)
                relu1 = layers.leaky_relu_layer(dense1)
                reshaped = tf.reshape(relu1, shape=[-1, 256, 5, 5]) # shape=(batch_size, 256, 5, 5)
                
            deconv1 = layers.deconv_block_fullres(reshaped, training, out_channels=128, filter_size=(3,3), strides=(1,1), padding="same", use_bias=False) # shape=(batch_size, 128, 5, 5)
            deconv2 = layers.deconv_block_fullres(deconv1, training, out_channels=64, filter_size=(5,5), strides=(5,5), padding="same", use_bias=False) # shape=(batch_size, 64, 25, 25)
            deconv3 = layers.deconv_block_fullres(deconv2, training, out_channels=32, filter_size=(3,3), strides=(1,1), padding="same", use_bias=False) # shape=(batch_size, 32, 25, 25)
            deconv4 = layers.deconv_block_fullres(deconv3, training, out_channels=16, filter_size=(5,5), strides=(5,5), padding="same", use_bias=False) # shape=(batch_size, 16, 125, 125)
            deconv5 = layers.deconv_block_fullres(deconv4, training, out_channels=8, filter_size=(3,3), strides=(2,2), padding="same", use_bias=False) # shape=(batch_size, 8, 250, 250)
            deconv6 = layers.deconv_block_fullres(deconv5, training, out_channels=4, filter_size=(3,3), strides=(2,2), padding="same", use_bias=False) # shape=(batch_size, 4, 500, 500)
            deconv7 = layers.deconv_layer(deconv6, out_channels=1, filter_size=(3,3), strides=(2,2), padding="same", use_bias=False) # shape=(batch_size, 1, 1000, 1000)
                
            gen_out = layers.tanh_layer(deconv7)
        print("Built Generator model in {} s".format(time.time()-a))
        list_ops = {"dense1": dense1, "batch1":batch1, "relu1": relu1, "reshaped": reshaped, "deconv1": deconv1, "deconv2": deconv2, 
                    "deconv3": deconv3, "deconv4": deconv4, "deconv5": deconv5, "deconv6": deconv6, "deconv7": deconv7, "gen_out": gen_out} # list of operations, can be used to run the graph up to a certain op
                                                                                                                     # i,e get the subgraph
        return gen_out, list_ops
        
    def discriminator_model(self, inp, training, reuse=False, minibatch=False): # construct the graph of the discriminator
        a = time.time()
        with tf.variable_scope("discriminator", reuse=reuse): # define variable scope to easily retrieve vars of the discriminator

            conv1 = layers.conv_block_fullres(inp, training, out_channels=4, filter_size=(3, 3), strides=(2, 2), padding="same", use_bias=False, alpha=0.3) # shape=(batch_size, 4, 500, 500)
            conv2 = layers.conv_block_fullres(conv1, training, out_channels=8, filter_size=(3, 3), strides=(2, 2), padding="same", use_bias=False, alpha=0.3) # shape=(batch_size, 256, 16, 16)
            conv3 = layers.conv_block_fullres(conv2, training, out_channels=16, filter_size=(3, 3), strides=(2, 2), padding="same", use_bias=False, alpha=0.3) # shape=(batch_size, 512, 8, 8)
            conv4 = layers.conv_block_fullres(conv3, training, out_channels=32, filter_size=(3, 3), strides=(2, 2), padding="same", use_bias=False, alpha=0.3) # shape=(batch_size, 1024, 4, 4)
            conv5 = layers.conv_block_fullres(conv4, training, out_channels=64, filter_size=(3, 3), strides=(2, 2), padding="same", use_bias=False, alpha=0.3) # shape=(batch_size, 256, 16, 16)
            conv6 = layers.conv_block_fullres(conv5, training, out_channels=128, filter_size=(3, 3), strides=(2, 2), padding="same", use_bias=False, alpha=0.3) # shape=(batch_size, 512, 8, 8)
            conv7 = layers.conv_block_fullres(conv6, training, out_channels=256, filter_size=(3, 3), strides=(2, 2), padding="same", use_bias=False, alpha=0.3) # shape=(batch_size, 1024, 4, 4)
            conv8 = layers.conv_block_fullres(conv7, training, out_channels=512, filter_size=(3, 3), strides=(2, 2), padding="same", use_bias=False, alpha=0.3) # shape=(batch_size, 1024, 4, 4)
            flat = tf.reshape(conv8, [-1, 512*4*4])
            dense1 = layers.dense_block_fullres(flat, training, units=4000)
            dense2 = layers.dense_block_fullres(dense1, training, units=500)

            if(minibatch):
                minibatched = layers.minibatch(dense2, num_kernels=5, kernel_dim=3)
                logits = layers.dense_layer(minibatched, units=2, use_bias=True)
            else:
                logits = layers.dense_layer(flat, units=2, use_bias=True)
            
            out = logits
        print("Built Discriminator model in {} s".format(time.time()-a))
        list_ops = {"inp": inp, "conv1": conv1, "conv2": conv2, "conv3": conv3, "conv4": conv4, "conv5": conv5, "conv6": conv6, "conv7": conv7, "flat": flat, "dense1": dense1, "dense2": dense2, "logits": logits, "out": out}
        
        return out, list_ops
        
    def generator_vars(self):
        gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")
        return gen_vars

    def discriminator_vars(self):
        discr_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")
        return discr_vars

    def generator_loss(self, fake_out, labels, label_smoothing=False):
        with tf.name_scope("generator_loss"):
            if(label_smoothing):
                smoothed_labels = tf.one_hot(labels, depth=2)
                loss_gen = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=smoothed_labels, logits=fake_out))
            else:
                loss_gen = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=fake_out))
        return loss_gen
        
    def discriminator_loss(self, fake_out, real_out, fake_labels, real_labels, label_smoothing=False):
        with tf.name_scope("discriminator_loss"):
            if(label_smoothing):
                delta = 0.1
                perturbation = tf.reshape(tf.constant(delta, shape=real_labels.shape), [-1, 1])
                added_perturbation = tf.concat([perturbation, -perturbation], axis=1)

                smoothed_fakes = tf.one_hot(fake_labels, depth=2)
                smoothed_reals = tf.clip_by_value(tf.one_hot(real_labels, depth=2) + added_perturbation, 0.0, 1.0)
                
                fake_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=smoothed_fakes, logits=fake_out)
                real_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=smoothed_reals, logits=real_out)
            else:
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


























