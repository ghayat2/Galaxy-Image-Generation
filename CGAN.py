import tensorflow as tf
import layers
import time

global_seed = 5

class CGAN:
    def __init__(self):
        return
    
    def generator_model(self, noise, y, training, reuse=False): # construct the graph of the generator
        a = time.time()
        
        with tf.variable_scope("generator", reuse=reuse): # define variable scope to easily retrieve vars of the generator
            
            gen_inp = tf.concat([noise, y], axis=-1)
            
            with tf.name_scope("processing_inp"):
                dense1 = layers.dense_layer(gen_inp, units=5*5*256)
                batch_norm1 = layers.batch_norm_layer(dense1, training, momentum=0.8)
                leaky_relu1 = layers.leaky_relu_layer(batch_norm1)
                reshaped = tf.reshape(leaky_relu1, [-1, 256, 5, 5])
            

            deconv1 = layers.deconv_block(reshaped, training, momentum=0.8, out_channels=128, filter_size=(3, 3), strides=(1, 1), padding='same', use_bias=False) # out_shape=(batch_size, 128, 5, 5)
            
            deconv2 = layers.deconv_block(deconv1, training, momentum=0.8, out_channels=64, filter_size=(5, 5), strides=(5, 5), padding='same', use_bias=False) # out_shape=(batch_size, 64, 25, 25)

            deconv3 = layers.deconv_block(deconv2, training, momentum=0.8, out_channels=32, filter_size=(3, 3), strides=(1, 1), padding='same', use_bias=False) # out_shape=(batch_size, 32, 25, 25)

            deconv4 = layers.deconv_block(deconv3, training, momentum=0.8, out_channels=16, filter_size=(5, 5), strides=(5, 5), padding='same', use_bias=False) # out_shape=(batch_size, 16, 125, 125)

            deconv5 = layers.deconv_block(deconv4, training, momentum=0.8, out_channels=8, filter_size=(3, 3), strides=(2, 2), padding='same', use_bias=False) # out_shape=(batch_size, 8, 250, 250)

            deconv6 = layers.deconv_block(deconv5, training, momentum=0.8, out_channels=4, filter_size=(3, 3), strides=(2, 2), padding='same', use_bias=False) # out_shape=(batch_size, 4, 500, 500)

            gen_out = layers.deconv_layer(deconv6, out_channels=1, filter_size=(3, 3), strides=(2, 2), padding='same', use_bias=False, activation="tanh") # out_shape=(batch_size, 1, 1000, 1000)
            
        print("Built Generator model in {} s".format(time.time()-a))
        list_ops = [dense1, batch_norm1, leaky_relu1, deconv1, deconv2, deconv3, deconv4, deconv5, deconv6, gen_out] # list of operations, can be used to run the graph up to a certain op
                                                                                                                     # i,e get the subgraph
        return gen_out, list_ops
        
    def discriminator_model(self, inp, y, training, reuse=False): # construct the graph of the discriminator
        a = time.time()
        with tf.variable_scope("discriminator", reuse=reuse): # define variable scope to easily retrieve vars of the discriminator
        
            y_image = tf.tile(tf.reshape(y, [-1, 1, 1, 1]), [1, 1, tf.shape(inp)[2], tf.shape(inp)[3]])
            discr_inp = tf.concat([inp, y_image], axis=1) # concat along channels dimension
            
            conv1 = layers.conv_block(discr_inp, training, dropout_rate=0.3, out_channels=4, filter_size=(3, 3), strides=(2, 2), padding='same', use_bias=True) # out_shape=(batch_size, 4, 500, 500)
            
            conv2 = layers.conv_block(conv1, training, dropout_rate=0.3, out_channels=8, filter_size=(3, 3), strides=(2, 2), padding='same', use_bias=True) # out_shape=(batch_size, 4, 500, 500)
            
            conv3 = layers.conv_block(conv2, training, dropout_rate=0.3, out_channels=16, filter_size=(3, 3), strides=(2, 2), padding='same', use_bias=True) # out_shape=(batch_size, 16, 125, 125)
            
            conv4 = layers.conv_block(conv3, training, dropout_rate=0.3, out_channels=32, filter_size=(5, 5), strides=(5, 5), padding='same', use_bias=True) # out_shape=(batch_size, 32, 25, 25)
            
            conv5 = layers.conv_block(conv4, training, dropout_rate=0.3, out_channels=64, filter_size=(3, 3), strides=(1, 1), padding='same', use_bias=True) # out_shape=(batch_size, 64, 25, 25)
            
            conv6 = layers.conv_block(conv5, training, dropout_rate=0.3, out_channels=128, filter_size=(5, 5), strides=(5, 5), padding='same', use_bias=True) # out_shape=(batch_size, 128, 5, 5)
            
            conv7 = layers.conv_block(conv6, training, dropout_rate=0.3, out_channels=256, filter_size=(3, 3), strides=(1, 1), padding='same', use_bias=True) # out_shape=(batch_size, 256, 5, 5)
            
            flat = tf.reshape(conv7, [tf.shape(conv7)[0], -1])
            
            dense_block1 = layers.dense_block(flat, training, units=4000, dropout_rate=0.3, use_bias=True) # apply dense_block: dense -> leakyReLU -> dropout
            
            dense_block2 = layers.dense_block(dense_block1, training, units=500, dropout_rate=0.3, use_bias=True) # apply dense_block: dense -> leakyReLU -> dropout
            
            logits = layers.dense_layer(dense_block2, units=2)
            
        print("Built Discriminator model in {} s".format(time.time()-a))
        list_ops = [conv1, conv2, conv3, conv4, conv5, conv6, conv7, flat, dense_block1, dense_block2, logits]
        
        return logits, list_ops
        
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

    def train_op(self, loss, learning_rate, var_list, scope):
        with tf.name_scope("train_op"):
            global_step = tf.Variable(0, name='global_step',trainable=False)
            
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope)
            with tf.control_dependencies(update_ops): # for batch_norm
                train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step, var_list)
        
        return train_op, global_step


























