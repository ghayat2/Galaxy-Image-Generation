import numpy as np
import os 
from datetime import datetime
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers, models, regularizers, optimizers, losses
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt

class Model:

    def __init__(self):
        #print("Data shape is: {}".format(data_shape))
        pass
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
        super(BaseModel, self).__init__()
        self.data_shape = data_shape
        self.noise_dim = noise_dim

        self.generator = self.make_generator_model()
        self.discriminator = self.make_discriminator_model()
        self.generator_optimizer = optimizers.Adam(1e-3)
        self.discriminator_optimizer = optimizers.Adam(1e-4)
        self.loss = losses.BinaryCrossentropy(from_logits=True)

        self.checkpoint = None
        self.checkpoint_dir = None
        self.checkpoint_prefix = None

        # Restore from lastest available checkpoint
        if reload_ckpt:
            self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                                  discriminator_optimizer=self.discriminator_optimizer,
                                                  generator=self.generator,
                                                  discriminator=self.discriminator)

            self.checkpoint_dir = checkpoint_dir
            self.checkpoint_prefix = checkpoint_prefix
            latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir) # returns None if no checkpoint found
            checkpoint_found = (latest_checkpoint is not None) # model can be restored if a checkpoint found

            if checkpoint_found:
                self.status = self.checkpoint.restore(latest_checkpoint)
                print(self.status)

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
    def __init__(self, latent_dim, learning_rate=1e-3, name="default_vae_name"):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.scoring = None
        self.__name__ = name

        self.inference_net = tf.keras.Sequential(
            [
              tf.keras.layers.InputLayer(input_shape=(1000, 1000, 1)),
              tf.keras.layers.Conv2D(
                  filters=32, kernel_size=20, strides=(5, 5), activation='relu', padding="same"),
              tf.keras.layers.Conv2D(
                  filters=32, kernel_size=10, strides=(5, 5), activation='relu', padding="same"),
              tf.keras.layers.Flatten(),
              # No activation
              tf.keras.layers.Dense(units=128, activation='relu'),
              tf.keras.layers.Dense(latent_dim + latent_dim)
            ]
        )

        self.generative_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=128, activation='relu'),
                tf.keras.layers.Dense(units=40*40*32, activation='relu'),
                tf.keras.layers.Reshape(target_shape=(40, 40, 32)),
                tf.keras.layers.Conv2DTranspose(
                  filters=32,
                  kernel_size=10,
                  strides=(5, 5),
                  padding="same",
                  activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                  filters=1,
                  kernel_size=20,
                  strides=(5, 5),
                  padding="same",
                  activation='linear') # no activation
            ]
        )

    def set_untrainable(self):
        for layer in self.generative_net.layers:
            layer.trainable = False
        for layer in self.inference_net.layers:
            layer.trainable = False

    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(1, self.latent_dim))
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

        def log_likelihood_loss(X_reconstruction, X_target):
            loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=X_reconstruction, labels=X_target)

            # Sum across channels and pixels, average across batch dimension
            return tf.reduce_mean(tf.reduce_sum(loss, axis=[1, 2, 3]), axis=0)

        def kld_loss(Z_mu, Z_logvar):
            # Return the KL divergence between Z and a Gaussian prior N(0, I)
            kld = -0.5 * tf.reduce_sum(1 + Z_logvar - Z_mu ** 2 - tf.exp(Z_logvar), axis=1)
            # Average across batch dimension
            return tf.reduce_mean(kld, axis=0)

        loss = log_likelihood_loss(x_logit, x) + kld_loss(mean, logvar)
        #print(f"ELBO: {-tf.reduce_mean(logpx_z + logpz - logqz_x)}, GAB_ELBO: {loss}")
        #print(f"log_like: {log_likelihood_loss(x_logit, x)}, kdl: {kld_loss(mean, logvar)}")
        return loss

    def compute_gradients(self, x):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)
        return tape.gradient(loss, self.trainable_variables), loss

    def apply_gradients(self, gradients):
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    def save_nets(self, dest_dir="./saved_models"):
        if not os.path.isdir(dest_dir):
            os.mkdir(dest_dir)
        self.generative_net.save(os.path.join(dest_dir, "generative_net_" + self.__name__))
        self.inference_net.save(os.path.join(dest_dir, "inference_net_" + self.__name__))
        if self.scoring is not None:
            self.scoring.save(os.path.join(dest_dir, "scoring_net_" + self.__name__))

    def load_nets(self, dest_dir="./saved_models"):
        self .generative_net.load_weights(os.path.join(dest_dir, "generative_net_" + self.__name__))
        self.inference_net.load_weights(os.path.join(dest_dir, "inference_net_" + self.__name__))
        if self.scoring is not None:
            self.scoring.load_weights(os.path.join(dest_dir, "scoring_net_" + self.__name__))

    def sample_and_show(self):
        x_sample = self.sample()
        pic = x_sample[0]
        plt.figure(figsize=(10, 10))
        plt.imshow(tf.squeeze(pic), cmap="gray")
        plt.show()

    def random_recon_and_show(self, X):
        i = tf.expand_dims(X[np.random.choice(X.shape[0])], axis=0)
        mean, logvar = self.encode(i)
        z = self.reparameterize(mean, logvar)
        i = self.decode(z, apply_sigmoid=True)
        plt.figure(figsize=(10, 10))
        plt.imshow(tf.squeeze(i), cmap="gray")
        plt.show()

    def compile_scoring(self, optimizer=None):
        self.scoring = tf.keras.models.clone_model(self.inference_net)
        self.make_scoring_model(self.scoring)
        optim = optimizer if optimizer is not None else self.optimizer
        self.scoring.compile(loss='mean_squared_error', optimizer=optim, metrics=["mse"])

    @staticmethod
    def make_scoring_model(model):
        model.pop()

        model.add(layers.Dense(100))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Dense(1))

        return model

    def set_name(self, name):
        self.__name__ = name


class VAEGAN(BaseModel):
    def __init__(self, vae, data_shape=None, noise_dim=None, checkpoint_dir=None, checkpoint_prefix=None,
                 reload_ckpt=False, name="VEAGAN_default"):
        if noise_dim is None:
            noise_dim = vae.inferense_net.output_shape
        if data_shape is None:
            data_shape = vae.inferense_net.output_shape
        self.vae = vae
        self.vae.set_untrainable()
        self.__name__ = name
        super(VAEGAN, self).__init__(data_shape, noise_dim, checkpoint_dir, checkpoint_prefix, reload_ckpt=reload_ckpt)


    def make_generator_model(self):
        #model = tf.keras.models.clone_model(self.vae.inference_net)
        model = keras.Sequential()

        model.add(layers.Dense(2*self.noise_dim, use_bias=False, input_shape=(self.noise_dim,)))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.LeakyReLU())

        model.add(layers.Dense(2 * self.noise_dim, use_bias=False))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.LeakyReLU())

        model.add(layers.Dense(self.noise_dim, use_bias=False))

        return model

    def make_discriminator_model(self):
        model = keras.Sequential()

        model.add(layers.Dense(self.noise_dim, use_bias=False, input_shape=(self.noise_dim,)))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Dense(self.noise_dim, use_bias=False))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Dense(1, use_bias=False))

        return model

    def save_nets(self, dest_dir="./saved_models"):
        if not os.path.isdir(dest_dir):
            os.mkdir(dest_dir)
        self.generator.save(os.path.join(dest_dir, "generator_" + self.__name__))
        self.discriminator.save(os.path.join(dest_dir, "discriminator_" + self.__name__))

    def load_nets(self, dest_dir="./saved_models"):
        self .generator.load_weights(os.path.join(dest_dir, "generator_" + self.__name__))
        self.discriminator.load_weights(os.path.join(dest_dir, "discriminator_" + self.__name__))


class TwoStepsVEAGAN(VAEGAN):
    def __init__(self, vae, data_shape=None, noise_dim=None, checkpoint_dir=None, checkpoint_prefix=None,
                 reload_ckpt=False, name="TwoStepsVEAGAN_default"):
        super(TwoStepsVEAGAN, self).__init__(vae, data_shape, noise_dim, checkpoint_dir, checkpoint_prefix,
                 reload_ckpt, name)
        self.generator_prime = self.make_generator_prime()
        self.discriminator_prime = self.make_discriminator_prime()

    def make_generator_prime(self):
        model = keras.Sequential()

        model.add(layers.Dense(5 * 5 * 256, use_bias=False, input_shape=(self.noise_dim/2,)))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.LeakyReLU())

        model.add(layers.Reshape((5, 5, 256)))
        assert model.output_shape == (None, 5, 5, 256)  # Note: None is the batch size

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

    def make_discriminator_prime(self):
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

    def save_nets(self, dest_dir="./saved_models"):
        super(TwoStepsVEAGAN, self).save_nets(dest_dir)
        self.generator_prime.save(os.path.join(dest_dir, "generator_prime_" + self.__name__))
        self.discriminator_prime.save(os.path.join(dest_dir, "discriminator_prime_" + self.__name__))

    def load_nets(self, dest_dir="./saved_models"):
        super(TwoStepsVEAGAN, self).load_nets(dest_dir)
        self.generator_prime.load_weights(os.path.join(dest_dir, "generator_prime_" + self.__name__))
        self.discriminator_prime.load_weights(os.path.join(dest_dir, "discriminator_prime_" + self.__name__))