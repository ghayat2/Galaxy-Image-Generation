import numpy as np
import os 
from datetime import datetime
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers, models, regularizers, optimizers, losses
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

class Model:

    def __init__(self, data_shape, noise_dim):
        #print("Data shape is: {}".format(data_shape))
        self.data_shape = data_shape 
        self.noise_dim = noise_dim

    def make_generator_model(self):
        raise NotImplementedError('subclasses must override make_generator_model()')

    def make_discriminator_model(self):
        raise NotImplementedError('subclasses must override make_discriminator_model()')

    def discriminator_loss(self, real_output, fake_output):
        raise NotImplementedError('subclasses must override discriminator_loss()')

    def generator_loss(self, fake_output):
        raise NotImplementedError('subclasses must override generator_loss()')

class BaseModel(Model):

    def __init__(self, data_shape, noise_dim, checkpoint_dir, checkpoint_prefix, reload_ckpt=False):
        #print("Data shape is: {}".format(data_shape))
        super(BaseModel, self).__init__(data_shape, noise_dim)

        self.generator = self.make_generator_model()
        self.discriminator = self.make_discriminator_model()
        self.generator_optimizer = optimizers.Adam(1e-3)
        self.discriminator_optimizer = optimizers.Adam(1e-4)
        self.loss = losses.BinaryCrossentropy(from_logits=True)

        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                 discriminator_optimizer=self.discriminator_optimizer,
                                 generator=self.generator,
                                 discriminator=self.discriminator)

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_prefix = checkpoint_prefix

        # Restore from lastest available checkpoint
        if reload_ckpt:
            latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir) # returns None if no checkpoint found
            checkpoint_found = (latest_checkpoint is not None) # model can be restored if a checkpoint found

            if checkpoint_found:
                status = self.checkpoint.restore(latest_checkpoint)

    def make_generator_model(self):
        model = keras.Sequential()

        model.add(layers.Dense(5*5*256, use_bias=False, input_shape=(self.noise_dim,)))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.LeakyReLU())

        model.add(layers.Reshape((5, 5, 256)))
        assert model.output_shape == (None, 5, 5, 256) # Note: None is the batch size
        
        model.add(layers.Conv2DTranspose(128, (3, 3), strides=(1, 1), padding='same', use_bias=False))
        assert model.output_shape == (None, 5, 5, 128)
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.LeakyReLU())
        
        model.add(layers.Conv2DTranspose(64, (5, 5), strides=(5, 5), padding='same', use_bias=False))
        assert model.output_shape == (None, 25, 25, 64)
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.LeakyReLU())
        
        model.add(layers.Conv2DTranspose(32, (3, 3), strides=(1, 1), padding='same', use_bias=False))
        assert model.output_shape == (None, 25, 25, 32)
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.LeakyReLU())
        
        model.add(layers.Conv2DTranspose(16, (5, 5), strides=(5, 5), padding='same', use_bias=False))
        assert model.output_shape == (None, 125, 125, 16)
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.LeakyReLU())
        
        model.add(layers.Conv2DTranspose(8, (3, 3), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 250, 250, 8)
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.LeakyReLU())
        
        model.add(layers.Conv2DTranspose(4, (3, 3), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 500, 500, 4)
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.LeakyReLU())
        
        model.add(layers.Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, 1000, 1000, 1)

        return model

    def make_discriminator_model(self):
        model = keras.Sequential()
      
        model.add(layers.Conv2D(4, (3, 3), strides=(2, 2), padding='same',
                                         input_shape=self.data_shape))
        assert model.output_shape == (None, 500, 500, 4)
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))
        
        model.add(layers.Conv2D(8, (3, 3), strides=(2, 2), padding='same'))
        assert model.output_shape == (None, 250, 250, 8)
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(16, (3, 3), strides=(2, 2), padding='same'))
        assert model.output_shape == (None, 125, 125, 16)
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(32, (5, 5), strides=(5, 5), padding='same'))
        assert model.output_shape == (None, 25, 25, 32)
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))
        
        model.add(layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
        assert model.output_shape == (None, 25, 25, 64)
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))
        
        model.add(layers.Conv2D(128, (5, 5), strides=(5, 5), padding='same'))
        assert model.output_shape == (None, 5, 5, 128)
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))
        
        model.add(layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same'))
        assert model.output_shape == (None, 5, 5, 256)
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))
        
        model.add(layers.Flatten())
        
        model.add(layers.Dense(4000))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))
        
        model.add(layers.Dense(500))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))
        
        model.add(layers.Dense(1))

        return model


    ## Removes the final layer(s) of the discriminator and adds complexity
    ## to try to predict scores on the learned representation from the
    ## galaxy classification

    def to_scoring(self, reload_ckpt=False):

        score_ckpt_file = self.checkpoint_dir + '/weights.best.hdf5'
        #Apparently there is an issue with the outputted loss/accuracy being better than the
        #evaluated one when using fit_generator, so just save last one for now (save_best_only = False)
        #https://github.com/keras-team/keras/issues/10014
        self.score_checkpoint = ModelCheckpoint(score_ckpt_file, monitor='loss', verbose=1, save_best_only=False, mode='max')
        self.score_callbacks = [self.score_checkpoint]

        logdir = os.path.join(
            "logs", datetime.now().strftime("%Y%m%d-%H%M%S")
        )

        tensorboard_callback = TensorBoard(
            log_dir=logdir, 
            write_graph=True
        )

        self.score_callbacks.append(tensorboard_callback)

        self.scorer = self.make_scoring_model(tf.keras.models.clone_model(self.discriminator))
        self.score_opt = optimizers.Adam(1e-4)

        if(reload_ckpt == True):
            self.scorer.load_weights(score_ckpt_file)

        self.scorer.compile(loss='mean_squared_error', optimizer=self.score_opt)

    def make_scoring_model(self, model):
        model.pop()

        model.add(layers.Dense(100))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Dense(1))

        return model

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.loss(tf.ones_like(real_output), real_output)
        fake_loss = self.loss(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        return self.loss(tf.ones_like(fake_output), fake_output)


class CVAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.optimizer = tf.keras.optimizers.Adam(1e-4)

        self.inference_net = tf.keras.Sequential(
            [
              tf.keras.layers.InputLayer(input_shape=(1000, 1000, 1)),
              tf.keras.layers.Conv2D(
                  filters=32, kernel_size=20, strides=(5, 5), activation='relu'),
              tf.keras.layers.Conv2D(
                  filters=32, kernel_size=10, strides=(5, 5), activation='relu'),
              tf.keras.layers.Flatten(),
              # No activation
              tf.keras.layers.Dense(latent_dim + latent_dim),
            ]
        )

        self.generative_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=40*40*32, activation=tf.nn.relu),
                tf.keras.layers.Dense(units=40*40*32, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(40, 40, 32)),
                tf.keras.layers.Conv2DTranspose(
                  filters=32,
                  kernel_size=10,
                  strides=(4, 4),
                  padding="SAME",
                  activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                  filters=1,
                  kernel_size=20,
                  strides=(4, 4),
                  padding="SAME") # no activation
            ]
        )

    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(10, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape) #sample from normal distribution
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.generative_net(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs

        return logits

    def log_normal_pdf(self, sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
            -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis)

    def compute_loss(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_logit = self.decode(z)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.reduce_mean(cross_ent, axis=[1, 2, 3])
        logpz = self.log_normal_pdf(z, 0., 0.)
        logqz_x = self.log_normal_pdf(z, mean, logvar)
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)

    def compute_gradients(self, x):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)
        return tape.gradient(loss, self.trainable_variables), loss

    def apply_gradients(self, gradients):
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
