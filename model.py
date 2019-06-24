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
        self.disc_dropout = 0.5
        self.gen_dropout = 0.2
        self.scoring_l2_regularisation = 1e-3

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
        model.add(layers.Dropout(self.gen_dropout))

        model.add(layers.Reshape((5, 5, 256)))
        assert model.output_shape == (None, 5, 5, 256) # Note: None is the batch size
        
        model.add(layers.Conv2DTranspose(128, (3, 3), strides=(1, 1), padding='same', use_bias=False))
        assert model.output_shape == (None, 5, 5, 128)
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(self.gen_dropout))

        model.add(layers.Conv2DTranspose(64, (5, 5), strides=(5, 5), padding='same', use_bias=False))
        assert model.output_shape == (None, 25, 25, 64)
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(self.gen_dropout))

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
        model.add(layers.Dropout(self.disc_dropout))
        
        model.add(layers.Conv2D(8, (3, 3), strides=(2, 2), padding='same'))
        assert model.output_shape == (None, 250, 250, 8)
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(self.disc_dropout))

        model.add(layers.Conv2D(16, (3, 3), strides=(2, 2), padding='same'))
        assert model.output_shape == (None, 125, 125, 16)
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(self.disc_dropout))

        model.add(layers.Conv2D(32, (5, 5), strides=(5, 5), padding='same'))
        assert model.output_shape == (None, 25, 25, 32)
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(self.disc_dropout))
        
        model.add(layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
        assert model.output_shape == (None, 25, 25, 64)
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(self.disc_dropout))
        
        model.add(layers.Conv2D(128, (5, 5), strides=(5, 5), padding='same'))
        assert model.output_shape == (None, 5, 5, 128)
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(self.disc_dropout))
        
        model.add(layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same'))
        assert model.output_shape == (None, 5, 5, 256)
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(self.disc_dropout))
        
        model.add(layers.Flatten())
        
        model.add(layers.Dense(4000))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(self.disc_dropout))
        
        model.add(layers.Dense(500))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(self.disc_dropout))
        
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
        #real_loss = self.loss(tf.ones_like(real_output), real_output)
        #fake_loss = self.loss(tf.zeros_like(fake_output), fake_output)
        real_loss = tf.reduce_mean(tf.math.squared_difference(real_output, 1.0))
        fake_loss = tf.reduce_mean(tf.square(fake_output))
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        #return self.loss(tf.ones_like(fake_output), fake_output)
        return tf.reduce_mean(tf.math.squared_difference(fake_output, 1.0))

class BaseFusedModel(Model):

    def __init__(self, data_shape, noise_dim, checkpoint_dir, checkpoint_prefix, reload_ckpt=False, dest_dir=None):
        #print("Data shape is: {}".format(data_shape))
        super(BaseFusedModel, self).__init__()
        self.labeler = self.make_labeling_model()
        self.labeler.load_weights(os.path.join(dest_dir, "labeler" + self.__name__))

        self.discriminator = self.fuse_models()

    def make_labeling_model(self):
        model = keras.Sequential()
        
        model.add(layers.Dense(20, activation='relu', input_shape=(self.feat_dim, )), trainable=False)
        model.add(layers.BatchNormalization(momentum=0.8), trainable=False)
        model.add(layers.Dropout(self.dropout), trainable=False)
        
        model.add(layers.Dense(10, activation='relu'), trainable=False)
        model.add(layers.BatchNormalization(momentum=0.8), trainable=False)
        model.add(layers.Dropout(self.dropout), trainable=False)

        model.add(layers.Dense(1), trainable=False)

        return model

    def fuse_models(self):
        combined = layers.concatenate([self.discrimiator.output, self.labeler.output])
        z = layers.Dense(1)(combined)

        return Model(inputs=[self.discrimiator.input, self.labeler.input], outputs=z)

    def save_nets(self, dest_dir="./saved_models"):
        if not os.path.isdir(dest_dir):
            os.mkdir(dest_dir)
        self.generator.save(os.path.join(dest_dir, "generator_" + self.__name__))
        self.discriminator.save(os.path.join(dest_dir, "discriminator_" + self.__name__))

    def load_nets(self, dest_dir="./saved_models"):
        self .generator.load_weights(os.path.join(dest_dir, "generator_" + self.__name__))
        self.discriminator.load_weights(os.path.join(dest_dir, "discriminator_" + self.__name__))





class ImageRegressor(BaseModel):
    def __init__(self, data_shape, noise_dim, checkpoint_dir, checkpoint_prefix, reload_ckpt=False):
        #print("Data shape is: {}".format(data_shape))
        super(ImageRegressor, self).__init__(data_shape, noise_dim, checkpoint_dir, checkpoint_prefix, reload_ckpt)

    def to_scoring(self, reload_ckpt=False):

        score_ckpt_file = self.checkpoint_dir + '/weights.best.hdf5'
        #Apparently there is an issue with the outputted loss/accuracy being better than the
        #evaluated one when using fit_generator, so just save last one for now (save_best_only = False)
        #https://github.com/keras-team/keras/issues/10014
        self.score_checkpoint = ModelCheckpoint(score_ckpt_file, monitor='val_loss', verbose=1, save_best_only=False, mode='max')
        self.score_callbacks = [self.score_checkpoint]

        logdir = os.path.join(
            "logs", datetime.now().strftime("%Y%m%d-%H%M%S")
        )

        tensorboard_callback = TensorBoard(
            log_dir=logdir, 
            write_graph=True
        )

        self.score_callbacks.append(tensorboard_callback)

        self.scorer = self.make_scoring_model(keras.Sequential())
        self.score_opt = optimizers.Adam(1e-4)

        if(reload_ckpt == True):
            self.scorer.load_weights(score_ckpt_file)

        self.scorer.compile(loss='mean_squared_error', optimizer=self.score_opt)

    def make_scoring_model(self):
        model = keras.Sequential()
        model.add(layers.Conv2D(4, (3, 3), strides=(2, 2), padding='same',
                                         input_shape=self.data_shape))
        assert model.output_shape == (None, 500, 500, 4)
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(self.disc_dropout))
        
        model.add(layers.Conv2D(8, (3, 3), strides=(2, 2), padding='same'))
        assert model.output_shape == (None, 250, 250, 8)
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(self.disc_dropout))

        model.add(layers.Conv2D(16, (3, 3), strides=(2, 2), padding='same'))
        assert model.output_shape == (None, 125, 125, 16)
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(self.disc_dropout))

        model.add(layers.Conv2D(32, (5, 5), strides=(5, 5), padding='same'))
        assert model.output_shape == (None, 25, 25, 32)
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(self.disc_dropout))
        
        model.add(layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
        assert model.output_shape == (None, 25, 25, 64)
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(self.disc_dropout))
        
        model.add(layers.Conv2D(128, (5, 5), strides=(5, 5), padding='same'))
        assert model.output_shape == (None, 5, 5, 128)
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(self.disc_dropout))
        
        model.add(layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same'))
        assert model.output_shape == (None, 5, 5, 256)
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(self.disc_dropout))
        
        model.add(layers.Flatten())
        
        model.add(layers.Dense(4096))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(self.disc_dropout))
        
        model.add(layers.Dense(1024))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(self.disc_dropout))

        model.add(layers.Dense(100))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Dense(1))

        return model

class CGAN(Model):
    def __init__(self, data_shape, noise_dim, checkpoint_dir=None, checkpoint_prefix=None, reload_ckpt=False,
                 p_galaxy=0.8):
        #print("Data shape is: {}".format(data_shape))
        super(CGAN, self).__init__()
        self.data_shape = data_shape
        self.noise_dim = noise_dim
        self.disc_dropout = 0.5
        self.gen_dropout = 0.2
        self.scoring_l2_regularisation = 1e-3

        self.generator = self.make_generator_model()
        self.discriminator = self.make_discriminator_model()
        self.generator_optimizer = optimizers.Adam(1e-3)
        self.discriminator_optimizer = optimizers.Adam(1e-4)
        self.loss = losses.BinaryCrossentropy(from_logits=True)

        self.checkpoint = None
        self.checkpoint_dir = None
        self.checkpoint_prefix = None
        self.p_galaxy = p_galaxy
        self.is_galaxy = False

    def make_generator_model(self):
        model = keras.Sequential()

        model.add(layers.Dense(4 * 4 * 1024, use_bias=False, input_shape=(self.noise_dim+1,)))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(self.gen_dropout))

        model.add(layers.Reshape((4, 4, 1024)))

        model.add(layers.Conv2DTranspose(512, (4, 4), strides=(2, 2), padding='same', use_bias=True))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(self.gen_dropout))

        model.add(layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same', use_bias=True))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', use_bias=True))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(1, (4, 4), strides=(2, 2), padding='same', use_bias=True))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.LeakyReLU())

        model.add(layers.Activation("tanh"))

        return model

    def make_discriminator_model(self):
        model = keras.Sequential()

        model.add(layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same', use_bias=True))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2D(256, (4, 4), strides=(2, 2), padding='same', use_bias=True))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2D(512, (4, 4), strides=(2, 2), padding='same', use_bias=True))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2D(1024, (4, 4), strides=(2, 2), padding='same', use_bias=True))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.LeakyReLU())

        model.add(layers.Flatten())

        model.add(layers.Dense(3))

        return model

    def discriminator_loss(self, real_output, fake_output, is_galaxy):
        # fake = 0, real galaxy = 2, real other = 1
        real_labels = tf.one_hot(tf.constant(is_galaxy+1, shape=(fake_output.shape[0], 1)), 3)
        real_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=real_labels, logits=real_output))
        fake_labels = tf.one_hot(tf.constant(0, shape=(fake_output.shape[0], 1)), 3)
        fake_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=fake_labels, logits=fake_output))
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output, is_galaxy):
        #return self.loss(tf.ones_like(fake_output), fake_output)
        labels = tf.one_hot(tf.constant(is_galaxy+1, shape=(fake_output.shape[0], 1)), 3)
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=fake_output))

class BetterD2GAN(Model):
    def __init__(self, data_shape, noise_dim, checkpoint_dir, checkpoint_prefix, reload_ckpt=False):
        #print("Data shape is: {}".format(data_shape))
        super(BetterD2GAN, self).__init__()
        self.data_shape = data_shape
        self.noise_dim = noise_dim
        self.disc_1_dropout = 0.2
        self.disc_2_dropout = 0.5
        self.gen_dropout = 0.2
        self.scoring_l2_regularisation = 1e-3

        self.generator = self.make_generator_model()
        self.discriminator_1 = self.make_discriminator_1_model()
        self.discriminator_2 = self.make_discriminator_2_model()
        self.generator_optimizer = optimizers.Adam(1e-3)
        self.discriminator_1_optimizer = optimizers.Adam(1e-4)
        self.discriminator_2_optimizer = optimizers.Adam(1e-4)
        self.loss = losses.BinaryCrossentropy(from_logits=True)

        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                 discriminator_1_optimizer=self.discriminator_1_optimizer,
                                 discriminator_2_optimizer=self.discriminator_2_optimizer,
                                 generator=self.generator,
                                 discriminator_1=self.discriminator_1,
                                 discriminator_2=self.discriminator_2)

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_prefix = checkpoint_prefix

        # Restore from lastest available checkpoint
        if reload_ckpt:
            latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir) # returns None if no checkpoint found
            checkpoint_found = (latest_checkpoint is not None) # model can be restored if a checkpoint found

            if checkpoint_found:
                self.status = self.checkpoint.restore(latest_checkpoint)
                print(self.status)

    def make_generator_model(self):
        model = keras.Sequential()

        model.add(layers.Dense(2*2*512, use_bias=False, input_shape=(self.noise_dim,)))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(self.gen_dropout))

        model.add(layers.Reshape((2, 2, 512)))
        assert model.output_shape == (None, 2, 2, 512) # Note: None is the batch size
        
        model.add(layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 4, 4, 256)
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(self.gen_dropout))

        # model.add(layers.Conv2DTranspose(16, (5, 5), strides=(5, 5), padding='same', use_bias=False))
        # assert model.output_shape == (None, 25, 25, 64)
        # model.add(layers.BatchNormalization(momentum=0.8))
        # model.add(layers.LeakyReLU())
        # model.add(layers.Dropout(self.gen_dropout))

        model.add(layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 8, 8, 128)
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.LeakyReLU())
        
        # model.add(layers.Conv2DTranspose(16, (5, 5), strides=(5, 5), padding='same', use_bias=False))
        # assert model.output_shape == (None, 125, 125, 16)
        # model.add(layers.BatchNormalization(momentum=0.8))
        # model.add(layers.LeakyReLU())
        
        model.add(layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 16, 16, 64)
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.LeakyReLU())
        
        model.add(layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 32, 32, 32)
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.LeakyReLU())
        
        model.add(layers.Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, 64, 64, 1)

        return model

    def make_discriminator_1_model(self):
        model = keras.Sequential()
      
        model.add(layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same',
                                         input_shape=self.data_shape))
        assert model.output_shape == (None, 32, 32, 32)
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(self.disc_1_dropout))
        
        model.add(layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same'))
        assert model.output_shape == (None, 16, 16, 64)
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(self.disc_1_dropout))

        model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
        assert model.output_shape == (None, 8, 8, 128)
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(self.disc_1_dropout))

        # model.add(layers.Conv2D(32, (5, 5), strides=(5, 5), padding='same'))
        # assert model.output_shape == (None, 25, 25, 32)
        # model.add(layers.LeakyReLU())
        # model.add(layers.Dropout(self.disc_1_dropout))
        
        model.add(layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
        assert model.output_shape == (None, 4, 4, 256)
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(self.disc_1_dropout))
        
        # model.add(layers.Conv2D(128, (5, 5), strides=(5, 5), padding='same'))
        # assert model.output_shape == (None, 5, 5, 128)
        # model.add(layers.LeakyReLU())
        # model.add(layers.Dropout(self.disc_1_dropout))
        
        model.add(layers.Conv2D(512, (3, 3), strides=(2, 2), padding='same'))
        assert model.output_shape == (None, 2, 2, 512)
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(self.disc_1_dropout))
        
        model.add(layers.Flatten())
        
        model.add(layers.Dense(1024))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(self.disc_1_dropout))
        
        model.add(layers.Dense(256))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(self.disc_1_dropout))
        
        model.add(layers.Dense(2, activation='softmax'))

        return model

    def make_discriminator_2_model(self):
        model = keras.Sequential()
      
        model.add(layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same',
                                         input_shape=self.data_shape))
        assert model.output_shape == (None, 32, 32, 32)
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(self.disc_2_dropout))
        
        model.add(layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same'))
        assert model.output_shape == (None, 16, 16, 64)
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(self.disc_2_dropout))

        model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
        assert model.output_shape == (None, 8, 8, 128)
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(self.disc_2_dropout))

        # model.add(layers.Conv2D(32, (5, 5), strides=(5, 5), padding='same'))
        # assert model.output_shape == (None, 25, 25, 32)
        # model.add(layers.LeakyReLU())
        # model.add(layers.Dropout(self.disc_1_dropout))
        
        model.add(layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
        assert model.output_shape == (None, 4, 4, 256)
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(self.disc_2_dropout))
        
        # model.add(layers.Conv2D(128, (5, 5), strides=(5, 5), padding='same'))
        # assert model.output_shape == (None, 5, 5, 128)
        # model.add(layers.LeakyReLU())
        # model.add(layers.Dropout(self.disc_1_dropout))
        
        model.add(layers.Conv2D(512, (3, 3), strides=(2, 2), padding='same'))
        assert model.output_shape == (None, 2, 2, 512)
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(self.disc_2_dropout))
        
        model.add(layers.Flatten())
        
        model.add(layers.Dense(1024))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(self.disc_2_dropout))
        
        model.add(layers.Dense(256))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(self.disc_2_dropout))
        
        model.add(layers.Dense(2, activation='softmax'))

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

        self.scorer = self.make_scoring_model(tf.keras.models.clone_model(self.discriminator_1))
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

    def discriminator_1_loss(self, real_output_1, fake_output_1):
        # real_loss = self.loss(tf.ones_like(real_output_1), real_output_1)
        # fake_loss = self.loss(tf.zeros_like(fake_output_1), fake_output_1)

        # 0 Fake, 1 Real

        ro1 = real_output_1[:, 0]
        fo1 = fake_output_1[:, 0]

        # print("ro1 shape: {}".format(ro1.shape))

        real_loss = 0.2 * -tf.math.reduce_sum(tf.math.log(real_output_1[:, 1]))
        fake_loss = tf.math.reduce_sum(fake_output_1[:, 1])

        # print("Real and fake 1: {} {}".format(real_loss, fake_loss))

        total_loss = real_loss + fake_loss
        return total_loss

    def discriminator_2_loss(self, real_output_2, fake_output_2):
        # real_loss = self.loss(tf.ones_like(real_output_2), real_output_2)
        # fake_loss = self.loss(tf.zeros_like(fake_output_2), fake_output_2)
        
        # 0 Real, 1 Fake    

        ro2 = real_output_2[:, 0]
        fo2 = fake_output_2[:, 0]

        # print("ro2: {}".format(real_output_2))      
        # print("fo2: {}".format(fake_output_2))      

        real_loss = tf.math.reduce_sum(real_output_2[:, 1])
        fake_loss = 0.1 * -tf.math.reduce_sum(tf.math.log(fake_output_2[:, 1]))

        # print("Real and fake 2: {} {}".format(real_loss, fake_loss))

        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output_1, fake_output_2):
        fo1 = fake_output_1[:, 0]
        fo2 = fake_output_2[:, 0]

        fake_loss_1 = tf.math.reduce_sum(fake_output_1[:, 0])
        fake_loss_2 = -0.1*tf.math.reduce_sum(tf.math.log(fake_output_2[:, 0]))

        # print("Fake Gen: {} {}".format(fake_loss_1, fake_loss_2))

        total_loss = fake_loss_1 + fake_loss_2
        return total_loss

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
              tf.keras.layers.Dense(units=256, activation='relu'),
              tf.keras.layers.Dense(latent_dim + latent_dim)
            ]
        )

        self.generative_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=256, activation='relu'),
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
        #print(f"ELBO: {}, GAB_ELBO: {loss}")
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
        self.generative_net.save_weights(os.path.join(dest_dir, "generative_net_" + self.__name__))
        self.inference_net.save_weights(os.path.join(dest_dir, "inference_net_" + self.__name__))
        if self.scoring is not None:
            self.scoring.save_weights(os.path.join(dest_dir, "scoring_net_" + self.__name__))

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

class POWERCVAE(CVAE):
    def __init__(self, latent_dim, data_shape=(1000, 1000, 1),learning_rate=1e-3, name="default_vae_name"):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.scoring = None
        self.__name__ = name
        self.data_shape = data_shape

        self.inference_net = self.make_encoder()
        self.generative_net = self.make_decoder()

    def make_decoder(self):
        model = keras.Sequential()

        model.add(layers.Dense(5 * 5 * 256, use_bias=False, input_shape=(self.latent_dim,)))
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

    def make_encoder(self):
        dropout_rate = 0.0
        model = keras.Sequential()

        model.add(layers.Conv2D(4, (3, 3), strides=(2, 2), padding='same',
                                input_shape=self.data_shape))
        assert model.output_shape == (None, 500, 500, 4)
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(dropout_rate))

        model.add(layers.Conv2D(8, (3, 3), strides=(2, 2), padding='same'))
        assert model.output_shape == (None, 250, 250, 8)
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(dropout_rate))

        model.add(layers.Conv2D(16, (3, 3), strides=(2, 2), padding='same'))
        assert model.output_shape == (None, 125, 125, 16)
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(dropout_rate))

        model.add(layers.Conv2D(32, (5, 5), strides=(5, 5), padding='same'))
        assert model.output_shape == (None, 25, 25, 32)
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(dropout_rate))

        model.add(layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
        assert model.output_shape == (None, 25, 25, 64)
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(dropout_rate))

        model.add(layers.Conv2D(128, (5, 5), strides=(5, 5), padding='same'))
        assert model.output_shape == (None, 5, 5, 128)
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(dropout_rate))

        model.add(layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same'))
        assert model.output_shape == (None, 5, 5, 256)
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(dropout_rate))

        model.add(layers.Flatten())

        model.add(layers.Dense(4000))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(dropout_rate))

        model.add(layers.Dense(self.latent_dim*2))

        return model

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

class LabelClassifier():
    def __init__(self, feat_dim, checkpoint_dir, checkpoint_prefix, reload_ckpt=False):
        super(LabelClassifier, self).__init__()
        self.dropout=0.5
        self.__name__ = "default_label_classifier"

        self.feat_dim = feat_dim

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_prefix = checkpoint_prefix

        self.labeler = self.make_labeling_model()
        self.optimizer = tf.keras.optimizers.Adam(1e-4)
        self.loss = losses.BinaryCrossentropy(from_logits=True)

        label_ckpt_file = self.checkpoint_dir + '/labeling_weights.best.hdf5'
        #Apparently there is an issue with the outputted loss/accuracy being better than the
        #evaluated one when using fit_generator, so just save last one for now (save_best_only = False)
        #https://github.com/keras-team/keras/issues/10014
        self.label_checkpoint = ModelCheckpoint(label_ckpt_file, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        self.label_callbacks = [self.label_checkpoint]

        logdir = os.path.join(
            "logs", datetime.now().strftime("%Y%m%d-%H%M%S")
        )

        self.label_opt = optimizers.Adam(1e-4)

        if(reload_ckpt == True):
            self.labeler.load_weights(label_ckpt_file)

        self.labeler.compile(loss='binary_crossentropy', optimizer=self.label_opt, metrics=['accuracy'])
    
    def set_name(self, name):
        self.__name__ = name

    def make_labeling_model(self):
        model = keras.Sequential()
        
        model.add(layers.Dense(20, activation='relu', input_shape=(self.feat_dim, )))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.Dropout(self.dropout))
        
        model.add(layers.Dense(10, activation='relu'))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.Dropout(self.dropout))

        model.add(layers.Dense(1))

        return model

    def apply_gradients(self, gradients):
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    def compute_loss(self, x, y):
        batch_size = x.shape[0]
        predicted = self.labeler(tf.convert_to_tensor(x, dtype=tf.float32))
        labels = tf.expand_dims(y, axis=1)
        loss = self.loss(labels, predicted)
        #print("y: {}".format(labels))
        #print("predicted: {}".format(tf.round(tf.nn.sigmoid(predicted))))
        num_correct = batch_size - np.sum(np.absolute(labels - tf.round(tf.nn.sigmoid(predicted))))
        #print("Num correct: {}".format(num_correct))
        return loss, num_correct

    def compute_gradients(self, x, y):
        with tf.GradientTape() as tape:
            loss, num_correct = self.compute_loss(x, y)
        return tape.gradient(loss, self.trainable_variables), loss, num_correct

    def save_nets(self, dest_dir="./saved_models"):
        if not os.path.isdir(dest_dir):
            os.mkdir(dest_dir)
        self.labeler.save(os.path.join(dest_dir, "labeler" + self.__name__))

    def load_nets(self, dest_dir="./saved_models"):
        self.labeler.load_weights(os.path.join(dest_dir, "labeler" + self.__name__))
