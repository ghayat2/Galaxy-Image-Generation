from datetime import datetime
import time
import numpy as np
import os
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.backend import set_session
from model import Model
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython import display


class Trainer:

    def __init__(
        self, model, sess, graph, train_dataset_labeled, train_dataset_scored, val_dataset_scored,
        out_path='../Results', verbose=True, debug=False
        ):

        self.model = model
        self.sess = sess
        self.graph = graph
        
        # Separate member for the tf.data type
        self.train_dataset_labeled = train_dataset_labeled
        self.train_dataset_scored = train_dataset_scored
        self.val_dataset_scored = val_dataset_scored

        self.fig_size = 20 # in inches
        self.num_examples = 16
        self.lines = np.sqrt(self.num_examples)
        self.cols = np.sqrt(self.num_examples)

        self.generate_every = 5

        self.callbacks = []
        self.verbose = verbose
        self.debug = debug

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


    def train(self, batch_size, seed, epochs=1, steps_per_epoch=2, save_every=15):
        step = 1
        gen_loss = -1
        disc_loss = -1
        for epoch in range(epochs):
            #print("Epoch: {}, Gen_loss: {}, Disc_loss: {}, step : {}".format(epoch, gen_loss, disc_loss, step))
            start = time.time()
            b = 0 # batch nb
            #iter = self.train_dataset_labeled.make_one_shot_iterator()
            for batch, labels in self.train_dataset_labeled:
                gen_loss, disc_loss = self.train_step(batch, batch_size)
                print("Epoch: {}, Batch: {}, Step: {}, Gen_loss: {}, Disc_loss: {}".format(epoch, b, step, gen_loss, disc_loss))
                b += 1
                if step % self.generate_every == 0:
                    display.clear_output(wait=True)
                    self.generate_and_save_images(seed, "step", nb = step)
                step += 1
                if(b >= steps_per_epoch):
                    break
            display.clear_output(wait=True)
            self.generate_and_save_images(seed, "epoch", nb = epoch)

            # Save the model every N epochs
            # NOTE: each checkpoint can be 400+ MB
            # If we checkpoint too much, it can cause serious trouble
            if (epoch + 1) % save_every == 0:
                self.model.checkpoint.save(file_prefix = self.model.checkpoint_prefix)

            print ('Time for epoch {} is {} sec'.format(epoch, time.time()-start))
        # Generate after the final epoch
        display.clear_output(wait=True)
        #self.generate_and_save_images(seed, "epoch", nb = epochs)

    def score(self, batch_size, epochs=10, steps_per_epoch=3):
        if(steps_per_epoch is not None):
            self.model.scorer.fit_generator(self.train_dataset_scored, steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=self.model.score_callbacks, verbose=1)
        else:
            self.model.scorer.fit_generator(self.train_dataset_scored, epochs=epochs, callbacks=self.model.score_callbacks, validation_data=self.val_dataset_scored, validation_steps=800, use_multiprocessing=True, verbose=1)

    def generate_and_save_images(self, test_input, str, nb):
        predictions = self.model.generator(test_input, training=False) # get generator output
        #print(predictions.shape)

        fig = plt.figure(figsize=(self.fig_size, self.fig_size)) # Create a new "fig_size" inches by "fig_size" inches figure as default figure

        #print("My Predictions Are: {}".format(predictions))
        #with self.graph.as_default():
        #    set_session(self.sess)
        for i in range(predictions.shape[0]):
            image = predictions[i, :, :, 0].numpy() # take the i'th predicted image, remove the last dimension (result is 2D)
            plt.subplot(self.lines, self.cols, i+1) # consider the default figure as lines x cols grid and select the (i+1)th cell
            plt.imshow(image, cmap='gray', vmin=-1.0, vmax=1.0) # plot the image on the selected cell
            plt.axis('off')
            maxval = image.max()
            minval = image.min()
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

    def vae_train(self, vae, epochs=75, steps_per_epoch=1000/32, show_sample=True):
        print(f"Steps per epochs = {steps_per_epoch}")
        for epoch in range(1, epochs + 1):
            self.dprint(f"epoch: {epoch}")
            start_time = time.time()
            b = 0
            for batch, labels in self.train_dataset_labeled:
                gradients, loss = vae.compute_gradients(batch)
                vae.apply_gradients(gradients)
                b += 1
                if b > steps_per_epoch:
                    break
            end_time = time.time()

            if epoch % self.generate_every == 0:
                loss = tf.keras.metrics.Mean()
                c = 0
                for test_batch, labels in self.train_dataset_labeled:
                    loss(vae.compute_loss(test_batch))
                    if c > steps_per_epoch:
                        break
                    c += 1
                elbo = -loss.result()
                self.vprint('Epoch: {}, Test set ELBO: {}'
                      'time elapse for current epoch {}'.format(epoch,
                                                                elbo,
                                                                end_time - start_time))
                if show_sample:
                    vae.sample_and_show()
        self.vprint(f"VAE trained for {epochs} epochs")

    def vprint(self, msg):
        if self.verbose:
            print(msg)
    def dprint(self, msg):
        if self.debug:
            print(msg)