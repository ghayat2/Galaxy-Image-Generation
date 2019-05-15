import numpy as np
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers, models, regularizers, optimizers


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

    def __init__(self, data_shape, noise_dim, checkpoint_dir, checkpoint_prefix):
        #print("Data shape is: {}".format(data_shape))
        super(BaseModel, self).__init__(data_shape, noise_dim)

        self.generator = self.make_generator_model()
        self.discriminator = self.make_discriminator_model()
        self.generator_optimizer = tf.compat.v1.train.AdamOptimizer(1e-3)
        self.discriminator_optimizer = tf.compat.v1.train.AdamOptimizer(1e-4)
        self.loss = tf.compat.v1.keras.losses.BinaryCrossentropy(from_logits=True)

        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                 discriminator_optimizer=self.discriminator_optimizer,
                                 generator=self.generator,
                                 discriminator=self.discriminator)

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_prefix = checkpoint_prefix

        # Restore from lastest available checkpoint
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir) # returns None if no checkpoint found
        checkpoint_found = (latest_checkpoint is not None) # model can be restored if a checkpoint found

        if checkpoint_found:
            status = checkpoint.restore(latest_checkpoint)

    def make_generator_model(self):
        model = tf.keras.Sequential()

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
        model = tf.keras.Sequential()
      
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

    def to_scoring(self):
        self.scorer = self.make_scoring_model(tf.keras.models.clone_model(self.discriminator))
        self.score_opt = tf.compat.v1.train.AdamOptimizer(1e-4)
        self.scorer.compile(loss='mean_squared_error', optimizer=self.score_opt)
        score_ckpt_file = self.checkpoint_dir + '/scoring-{epoch:02d}-{val_acc:.2f}.hdf5'
        self.score_checkpoint = ModelCheckpoint(score_ckpt_file, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        self.score_callbacks = [self.score_checkpoint]

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

#class GAN_Generator(Model):







