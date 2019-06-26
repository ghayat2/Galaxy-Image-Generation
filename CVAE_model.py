import tensorflow as tf
import layers
import time, sys

global_seed = 5

class CVAE:
    def __init__(self, batch_size, latent_dim):
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        return
    
    def inference_model(self, inp, training, reuse=False): # construct the graph of the inference net
        a = time.time()
        
        with tf.variable_scope("inference", reuse=reuse): # define variable scope
            conv1 = layers.conv_layer(inp, out_channels=32, filter_size=(20, 20), strides=(5, 5), 
                                      padding="same", use_bias=True, activation=None)
            relu1 = layers.relu_layer(conv1)
            
            conv2 = layers.conv_layer(relu1, out_channels=32, filter_size=(10, 10), strides=(5, 5), 
                                      padding="same", use_bias=True, activation=None)
            relu2 = layers.relu_layer(conv2)
            
            flat = tf.reshape(relu2, [self.batch_size, -1])
            
            dense1 = layers.dense_layer(flat, units=256, use_bias=True)
            relu3 = layers.relu_layer(dense1)
            dense2 = layers.dense_layer(relu3, units=2*self.latent_dim, use_bias=True)
            
            mean, logvar = tf.split(dense2, num_or_size_splits=2, axis=1)
#        print(logvar.shape)
#        sys.exit(0)
            
#            self.inference_net = tf.keras.Sequential(
#            [
#              tf.keras.layers.InputLayer(input_shape=(1000, 1000, 1)),
#              tf.keras.layers.Conv2D(
#                  filters=32, kernel_size=20, strides=(5, 5), activation='relu', padding="same"),
#              tf.keras.layers.Conv2D(
#                  filters=32, kernel_size=10, strides=(5, 5), activation='relu', padding="same"),
#              tf.keras.layers.Flatten(),
#              # No activation
#              tf.keras.layers.Dense(units=256, activation='relu'),
#              tf.keras.layers.Dense(latent_dim + latent_dim)
#            ]
#        )
                
        print("Built Inference model in {} s".format(time.time()-a))
        list_ops = [conv1, relu1, conv2, relu2, flat, dense1, relu3, dense2] # list of operations, can be used to run the graph up to a certain op
                                                                                                                     # i,e get the subgraph
        return mean, logvar, list_ops
        
    def generative_model(self, noise, training, reuse=False, apply_sigmoid=False): # construct the graph of the generative net
        a = time.time()
        
        with tf.variable_scope("generative", reuse=reuse): # define variable scope to easily retrieve vars of the discriminator
            dense1 = layers.dense_layer(noise, units=256, use_bias=True)
            relu1 = layers.relu_layer(dense1)
            dense2 = layers.dense_layer(relu1, units=40*40*32, use_bias=True)
            relu2 = layers.relu_layer(dense2)
            
            reshaped = tf.reshape(relu2, [-1, 32, 40, 40])
            
            deconv1 = layers.deconv_layer(reshaped, out_channels=32, filter_size=(10, 10), strides=(5, 5), 
                                          padding="same", use_bias=True, activation=None)
            relu3 = layers.relu_layer(deconv1)
            deconv2 = layers.deconv_layer(relu3, out_channels=1, filter_size=(20, 20), strides=(5, 5), 
                                          padding="same", use_bias=True, activation=None)
        logits = deconv2
        
        out = tf.sigmoid(logits)
#        print(relu3.shape)
#        sys.exit(0)
        
#            self.generative_net = tf.keras.Sequential(
#            [
#                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
#                tf.keras.layers.Dense(units=256, activation='relu'),
#                tf.keras.layers.Dense(units=40*40*32, activation='relu'),
#                tf.keras.layers.Reshape(target_shape=(40, 40, 32)),
#                tf.keras.layers.Conv2DTranspose(
#                  filters=32,
#                  kernel_size=10,
#                  strides=(5, 5),
#                  padding="same",
#                  activation='relu'),
#                tf.keras.layers.Conv2DTranspose(
#                  filters=1,
#                  kernel_size=20,
#                  strides=(5, 5),
#                  padding="same",
#                  activation='linear') # no activation
#            ]
#        )
        print("Built Generative model in {} s".format(time.time()-a))
        list_ops = [dense1, relu1, dense2, relu2, reshaped, deconv1, relu3, deconv2, logits]
        
        return logits, out, list_ops
    
    def reparameterize(self, mean, logvar):
        with tf.name_scope("reparametrization"):
            eps = tf.random.normal(shape=mean.shape) #sample from normal distribution
            return eps * tf.exp(logvar * .5) + mean

    def compute_loss(self, mean, logvar, logits, labels, z):

        def log_likelihood_loss(logits, labels):
            with tf.name_scope("log_likelihood_loss"):
                loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
                # Sum across channels and pixels, average across batch dimension
                return tf.reduce_mean(tf.reduce_sum(loss, axis=[1, 2, 3]), axis=0)

        def kld_loss(Z_mu, Z_logvar):
            with tf.name_scope("kld_loss"):
                # Return the KL divergence between Z and a Gaussian prior N(0, I)
                kld = -0.5 * tf.reduce_sum(1 + Z_logvar - Z_mu ** 2 - tf.exp(Z_logvar), axis=1)
                # Average across batch dimension
                return tf.reduce_mean(kld, axis=0)
        
        with tf.name_scope("loss"):
            loss = log_likelihood_loss(logits, labels) + kld_loss(mean, logvar)
            return loss

    def train_op(self, loss, learning_rate, beta1, beta2):
        with tf.name_scope("train_op"):
            global_step = tf.Variable(0, name='global_step',trainable=False)
            
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops): # for batch_norm
                train_op = tf.train.AdamOptimizer(learning_rate, beta1, beta2).minimize(loss, global_step)
        
        return train_op, global_step


























