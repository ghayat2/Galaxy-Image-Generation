from datetime import datetime
import time
import numpy as np
import os
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint
from model import Model
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython import display



class Trainer:

    def __init__(
        self, model, train_dataset_labeled, train_dataset_scored,
        out_path='../Results'
        ):

        self.model = model
        
        # Separate member for the tf.data type
        self.train_dataset_labeled = train_dataset_labeled
        self.train_dataset_scored = train_dataset_scored

        self.fig_size = 20 # in inches
        self.num_examples = 16
        self.lines = np.sqrt(self.num_examples)
        self.cols = np.sqrt(self.num_examples)

        self.generate_every = 3

        self.callbacks = []

        # Output path
        self.out_path = out_path
        if not os.path.exists(out_path):
            os.makedirs(out_path)

    def train_step(self, images, batch_size):
        noise = tf.random.normal([batch_size, self.model.noise_dim]) # number of generated images equal to number of real images provided
                                                          # to discriminator (i,e batch_size)

        #images = tf.expand_dims(images, 3)
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.model.generator(noise, training=True)

            real_output = self.model.discriminator(images, training=True)
            fake_output = self.model.discriminator(generated_images, training=True)

            gen_loss = self.model.generator_loss(fake_output)
            disc_loss = self.model.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.model.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.model.discriminator.trainable_variables)

        self.model.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.model.generator.trainable_variables))
        self.model.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.model.discriminator.trainable_variables))
        return gen_loss, disc_loss


    def train(self, batch_size, seed, epochs=1):
        step = 1
        gen_loss = -1
        disc_loss = -1
        for epoch in range(epochs):
            print("Epoch: {}, Gen_loss: {}, Disc_loss: {}, step : {}".format(epoch, gen_loss, disc_loss, step))
            start = time.time()
            b = 0 # batch nb
            #iter = self.train_dataset_labeled.make_one_shot_iterator()
            for batch in self.train_dataset_labeled.batch(batch_size):
                gen_loss, disc_loss = self.train_step(batch, batch_size)
                #print("Epoch: {}, Batch: {}, Step: {}, Gen_loss: {}, Disc_loss: {}".format(epoch, b, step, gen_loss, disc_loss))
                b += 1
                #if step % self.generate_every == 0:
                    #display.clear_output(wait=True)
                    #self.generate_and_save_images(seed, "step", nb = step)
                step += 1
            display.clear_output(wait=True)
            self.generate_and_save_images(seed, "epoch", nb = epoch)

            # Save the model every 15 epochs
            if (epoch + 1) % 15 == 0:
                self.model.checkpoint.save(file_prefix = self.model.checkpoint_prefix)

            print ('Time for epoch {} is {} sec'.format(epoch, time.time()-start))
        # Generate after the final epoch
        display.clear_output(wait=True)
        self.generate_and_save_images(seed, "epoch", nb = epochs)

    def score(self, batch_size, epochs=1):

        self.model.scorer.fit_generator(self.train_dataset_scored, epochs=epochs, callbacks=self.model.score_callbacks)

    def generate_and_save_images(self, test_input, str, nb):
        predictions = self.model.generator(test_input, training=False) # get generator output
        #print(predictions.shape)

        fig = plt.figure(figsize=(self.fig_size, self.fig_size)) # Create a new "fig_size" inches by "fig_size" inches figure as default figure

        #print("My Predictions Are: {}".format(predictions))

        for i in range(predictions.shape[0]):
            image = predictions[i, :, :, 0] # take the i'th predicted image, remove the last dimension (result is 2D)
            plt.subplot(self.lines, self.cols, i+1) # consider the default figure as lines x cols grid and select the (i+1)th cell
            plt.imshow(image.numpy(), cmap='gray', vmin=-1.0, vmax=1.0) # plot the image on the selected cell
            plt.axis('off')
            maxval = image.numpy().max()
            minval = image.numpy().min()
        print('Max and min vals: {} {}'.format(maxval, minval))
    #    plt.show() # finished plotting all images in the figure so show default figure

        if not os.path.exists(self.out_path): # create images dir if not existant
            os.mkdir(self.out_path)
        save_file = self.out_path + "/image_after_{}_{}.png".format(str, nb)
        print(save_file)
        plt.savefig(save_file) # save image to dir
        return predictions
    
    #### Kept from project for now ###
    def create_callbacks(self):
        """ Saves a  tensorboard with evolution of the loss """

        logdir = os.path.join(
            "logs", datetime.now().strftime("%Y%m%d-%H%M%S")
        )

        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir=logdir, 
            write_graph=True
        )

        self.callbacks.append(tensorboard_callback)

    ### -------------- ###

