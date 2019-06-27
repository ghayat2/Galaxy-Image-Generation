import tensorflow as tf
import layers
import time, sys

global_seed = 5

class CVAE:
    def __init__(self, batch_size, latent_dim):
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        return
    
    def inference_model(self, inp, training, reuse=False, resize=True): # construct the graph of the inference net
        a = time.time()
        
        with tf.variable_scope("inference", reuse=reuse): # define variable scope
            if resize:
                inp = layers.max_pool_layer(inp, pool_size=(2,2), strides=(2,2), padding=(12,12))
                inp = layers.max_pool_layer(inp, pool_size=(2,2), strides=(2,2))
                inp = layers.max_pool_layer(inp, pool_size=(2,2), strides=(2,2))
                inp = layers.max_pool_layer(inp, pool_size=(2,2), strides=(2,2))
                inp = layers.max_pool_layer(inp, pool_size=(2,2), strides=(2,2))
            
            flat = tf.reshape(inp, [self.batch_size, -1])
            dense1 = layers.dense_layer(flat, units=1024, use_bias=True)
            relu1 = tf.nn.softplus(dense1) # <-------------------------- maybe try softplus
            dense2 = layers.dense_layer(relu1, units=512, use_bias=True)
            relu2 = tf.nn.softplus(dense2)
            dense3 = layers.dense_layer(relu2, units=512, use_bias=True)
            relu3 = tf.nn.softplus(dense3)
            dense4 = layers.dense_layer(relu3, units=2*self.latent_dim, use_bias=True)
            
            mean, logvar = tf.split(dense4, num_or_size_splits=2, axis=1)
            
        print("Built Inference model in {} s".format(time.time()-a))
        list_ops = [flat, dense1, relu1, dense2, relu2, dense3, relu3, dense4] # list of operations, can be used to run the graph up to a certain op
                                                                                                                     # i,e get the subgraph
        return inp, mean, logvar, list_ops
        
    def generative_model(self, noise, training, reuse=False): # construct the graph of the generative net
        a = time.time()
        
        with tf.variable_scope("generative", reuse=reuse): # define variable scope to easily retrieve vars of the discriminator
        
            gen_inp = noise
            dense1 = layers.dense_layer(noise, units=512, use_bias=True)
            relu1 = tf.nn.softplus(dense1) # <-------------------------- maybe try softplus
            dense2 = layers.dense_layer(relu1, units=512, use_bias=True)
            relu2 = tf.nn.softplus(dense2)
            dense3 = layers.dense_layer(relu2, units=1024, use_bias=True)
            relu3 = tf.nn.softplus(dense3)
            dense4 = layers.dense_layer(relu3, units=1024, use_bias=True)
            
            reshaped = tf.reshape(dense4, [-1, 1, 32, 32])

            logits = reshaped
            out = tf.sigmoid(logits)
        print("Built Generative model in {} s".format(time.time()-a))
        list_ops = [dense1, relu1, dense2, relu2, dense3, relu3, dense4, reshaped]
        
        return logits, out, list_ops
    
    def reparameterize(self, mean, logvar):
        with tf.name_scope("reparametrization"):
            eps = tf.random.normal(shape=mean.shape, seed=global_seed) #sample from normal distribution
            return eps * tf.exp(logvar * .5) + mean

    def compute_loss(self, mean, logvar, logits, labels, z):

        def log_likelihood_loss(logits, labels):
            with tf.name_scope("log_likelihood_loss"):
                flat_logits = tf.reshape(logits, [self.batch_size, -1])
                flat_labels = tf.reshape(labels, [self.batch_size, -1])
#                print(flat_labels.shape)
#                print(flat_logits.shape)
                loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=flat_logits, labels=flat_labels)
                # Sum across pixels
                loss = tf.reduce_sum(loss, axis=1)
#                print(loss.shape)
                return loss
#                loss = tf.square(flat_labels - tf.sigmoid(flat_logits))
#                # Sum across pixels
#                loss = tf.reduce_mean(loss, axis=1)
##                print(loss.shape)

        def kld_loss(Z_mu, Z_logvar):
            with tf.name_scope("kld_loss"):
                # Return the KL divergence between q(Z|X) and a Gaussian prior N(0, I)
                kld = -0.5 * tf.reduce_sum(1 + Z_logvar - tf.square(Z_mu) - tf.exp(Z_logvar), axis=1)
#                print(kld.shape)
                return kld
        
        with tf.name_scope("loss"):
            loss = tf.reduce_mean(log_likelihood_loss(logits, labels) + kld_loss(mean, logvar), axis=0)
            return loss

    def train_op(self, loss, learning_rate, beta1, beta2):
        with tf.name_scope("train_op"):
            global_step = tf.Variable(0, name='global_step',trainable=False)
            
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops): # for batch_norm
                train_op = tf.train.AdamOptimizer(learning_rate, beta1, beta2).minimize(loss, global_step)
        
        return train_op, global_step


























