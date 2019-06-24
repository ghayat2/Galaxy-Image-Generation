from datetime import datetime
import time
import numpy as np
import os
import tensorflow as tf
from model import Model
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython import display

from sklearn.datasets import load_sample_image
from sklearn.feature_extraction import image
from skimage.feature import blob_doh, blob_log
from skimage.exposure import histogram
from skimage.feature import shape_index
from skimage.measure import shannon_entropy


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

        self.generate_every = 1

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

    def cgan_train_step(self, images, batch_size, is_galaxy):
        noise = tf.random.normal([batch_size, self.model.noise_dim]) # number of generated images equal to number of real images provided
                                                          # to discriminator (i,e batch_size)
        c = tf.constant(is_galaxy, shape=(batch_size, 1), dtype=tf.float32)
        input = tf.concat([noise, c], axis=-1)

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.model.generator(input, training=True)
            real_output = self.model.discriminator(images, training=True)
            fake_output = self.model.discriminator(generated_images, training=True)

            gen_loss = self.model.generator_loss(fake_output, is_galaxy)
            disc_loss = self.model.discriminator_loss(real_output, fake_output, is_galaxy)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.model.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.model.discriminator.trainable_variables)

        self.model.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.model.generator.trainable_variables))
        self.model.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.model.discriminator.trainable_variables))
        return gen_loss, disc_loss

    def d2gan_train_step(self, images, batch_size):
        noise = tf.random.normal([batch_size, self.model.noise_dim]) # number of generated images equal to number of real images provided
                                                          # to discriminator (i,e batch_size)
        #images = tf.expand_dims(images, 3)
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_1_tape, tf.GradientTape() as disc_2_tape:
            generated_images = self.model.generator(noise, training=True)

            real_output_1 = self.model.discriminator_1(images, training=True)
            fake_output_1 = self.model.discriminator_1(generated_images, training=True)

            real_output_2 = self.model.discriminator_2(images, training=True)
            fake_output_2 = self.model.discriminator_2(generated_images, training=True)

            gen_loss = self.model.generator_loss(fake_output_1, fake_output_2)
            disc_1_loss = self.model.discriminator_1_loss(real_output_1, fake_output_1)
            disc_2_loss = self.model.discriminator_2_loss(real_output_2, fake_output_2)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.model.generator.trainable_variables)
        gradients_of_discriminator_1 = disc_1_tape.gradient(disc_1_loss, self.model.discriminator_1.trainable_variables)
        gradients_of_discriminator_2 = disc_2_tape.gradient(disc_2_loss, self.model.discriminator_2.trainable_variables)

        self.model.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.model.generator.trainable_variables))
        self.model.discriminator_1_optimizer.apply_gradients(zip(gradients_of_discriminator_1, self.model.discriminator_1.trainable_variables))
        self.model.discriminator_2_optimizer.apply_gradients(zip(gradients_of_discriminator_2, self.model.discriminator_2.trainable_variables))
        return gen_loss, disc_1_loss, disc_2_loss

    def train_step_prime(self, images, batch_size):
        noise = tf.random.normal([batch_size, self.model.noise_dim]) # number of generated images equal to number of real images provided
                                                          # to discriminator (i,e batch_size)
        #images = tf.expand_dims(images, 3)
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            mean, logvar = tf.split(self.model.generator(noise), num_or_size_splits=2, axis=1)
            z = self.model.vae.reparameterize(mean, logvar)

            generated_images = self.model.generator_prime(z, training=True)
            real_output = self.model.discriminator_prime(images, training=True)
            fake_output = self.model.discriminator_prime(generated_images, training=True)

            gen_loss = self.model.generator_loss(fake_output)
            disc_loss = self.model.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.model.generator_prime.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.model.discriminator_prime.trainable_variables)

        self.model.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.model.generator_prime.trainable_variables))
        self.model.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.model.discriminator_prime.trainable_variables))
        return gen_loss, disc_loss


    def train(self, batch_size, seed, epochs=1, steps_per_epoch=2, save_every=15, batch_processing_fct=None, gen_imgs=True):
        step = 1
        gen_loss = -1
        disc_loss = -1
        for epoch in range(epochs):
            print("Epoch: {}, Gen_loss: {}, Disc_loss: {}, step : {}".format(epoch, gen_loss, disc_loss, step))
            start = time.time()
            b = 0 # batch nb
            #iter = self.train_dataset_labeled.make_one_shot_iterator()
            for batch, labels in self.train_dataset_labeled:
                if batch_processing_fct is not None:
                    batch = tf.squeeze(batch_processing_fct(batch))
                gen_loss, disc_loss = self.train_step(batch, batch_size)
                #print("Epoch: {}, Batch: {}, Step: {}, Gen_loss: {}, Disc_loss: {}".format(epoch, b, step, gen_loss, disc_loss))
                # gen_loss, disc_1_loss, disc_2_loss = self.d2gan_train_step(batch, batch_size)
                b += 1
                if step % self.generate_every == 0:
                    display.clear_output(wait=False)
                    if gen_imgs:
                        self.generate_and_save_images(seed, "step", nb = step, show=True)
                step += 1
                if(b >= steps_per_epoch):
                    break
            if gen_imgs:
                self.generate_and_save_images(seed, "epoch", nb = epoch, show=True)

            # Save the model every N epochs
            # NOTE: each checkpoint can be 400+ MB
            # If we checkpoint too much, it can cause serious trouble
            if (epoch + 1) % save_every == 0:
                # self.model.checkpoint.save(file_prefix = self.model.checkpoint_prefix)
                self.model.save_nets()

            print ('Time for epoch {} is {} sec'.format(epoch, time.time()-start))
        # Generate after the final epoch
        display.clear_output(wait=False)
        self.generate_and_save_images(seed, "epoch", nb = epochs)

    def vaegan_train(self, batch_size, seed, epochs=1, steps_per_epoch=2, save_every=15, batch_processing_fct=None,
                     gen_imgs=True, save_ckpt=False):
        step = 1
        gen_loss = -1
        disc_loss = -1
        for epoch in range(epochs):
            #print("Epoch: {}, Gen_loss: {}, Disc_loss: {}, step : {}".format(epoch, gen_loss, disc_loss, step))
            start = time.time()
            b = 0 # batch nb
            #iter = self.train_dataset_labeled.make_one_shot_iterator()
            for batch, labels in self.train_dataset_labeled:
                if batch_processing_fct is not None:
                    batch = tf.squeeze(batch_processing_fct(batch))
                gen_loss, disc_loss = self.train_step(batch, batch_size)
                #print("Epoch: {}, Batch: {}, Step: {}, Gen_loss: {}, Disc_loss: {}".format(epoch, b, step, gen_loss, disc_loss))
                b += 1
                if step % self.generate_every == 0:
                    #display.clear_output(wait=True)
                    if gen_imgs:
                        self.generate_and_save_images(seed, "step", nb = step, vae=True, training_id="veagan")
                step += 1
                if(b >= steps_per_epoch):
                    break
            #display.clear_output(wait=True)
            if gen_imgs:
                display.clear_output(wait=True)
                self.generate_and_save_images(seed, "epoch", nb = epoch, vae=True, show=True, training_id="veagan")

            # Save the model every N epochs
            # NOTE: each checkpoint can be 400+ MB
            # If we checkpoint too much, it can cause serious trouble
            if save_ckpt and (epoch + 1) % save_every == 0:
                self.model.checkpoint.save(file_prefix = self.model.checkpoint_prefix)

            display.clear_output(wait=True)
            print ('Time for epoch {} is {} sec'.format(epoch, time.time()-start))
            print("Epoch: {}, Batch: {}, Step: {}, Gen_loss: {}, Disc_loss: {}".format(epoch, b, step, gen_loss, disc_loss))

        # Generate after the final epoch
        #display.clear_output(wait=True)
        self.generate_and_save_images(seed, "epoch", nb = epochs, vae=True, show=True, training_id="veagan")

    def two_steps_vaegan_train(self, batch_size, seed, epochs=1, steps_per_epoch=2, save_every=15, batch_processing_fct=None,
                     gen_imgs=True, save_ckpt=False):
        step = 1
        gen_loss = -1
        disc_loss = -1
        for epoch in range(epochs):
            #print("Epoch: {}, Gen_loss: {}, Disc_loss: {}, step : {}".format(epoch, gen_loss, disc_loss, step))
            start = time.time()
            b = 0 # batch nb
            #iter = self.train_dataset_labeled.make_one_shot_iterator()
            for batch, labels in self.train_dataset_labeled:
                if batch_processing_fct is not None:
                    batch = tf.squeeze(batch_processing_fct(batch))
                gen_loss, disc_loss = self.train_step_prime(batch, batch_size)
                #print("Epoch: {}, Batch: {}, Step: {}, Gen_loss: {}, Disc_loss: {}".format(epoch, b, step, gen_loss, disc_loss))
                b += 1
                # if step % self.generate_every == 0:
                #     #display.clear_output(wait=True)
                #     if gen_imgs:
                #         self.generate_and_save_images(seed, "step", nb = step, vae=None, training_id="2steps", prime=True)
                step += 1
                if(b >= steps_per_epoch):
                    break
            #display.clear_output(wait=True)
            if gen_imgs:
                self.generate_and_save_images(seed, "epoch", nb = epoch, vae=None, show=False, training_id="2steps", prime=True)

            # Save the model every N epochs
            # NOTE: each checkpoint can be 400+ MB
            # If we checkpoint too much, it can cause serious trouble
            if save_ckpt and (epoch + 1) % save_every == 0:
                self.model.checkpoint.save(file_prefix = self.model.checkpoint_prefix)

            print ('Time for epoch {} is {} sec'.format(epoch, time.time()-start))
            print("Epoch: {}, Batch: {}, Step: {}, Gen_loss: {}, Disc_loss: {}".format(epoch, b, step, gen_loss, disc_loss))

        # Generate after the final epoch
        #display.clear_output(wait=True)
        self.generate_and_save_images(seed, "epoch", nb = epochs, vae=None, show=False, training_id="2steps", prime=True)

    def cgan_train(self, batch_size, seed, galaxy_generator, other_generator,
                   epochs=1, steps_per_epoch_galaxy=2, steps_per_epoch_other=2, save_every=15,
                   batch_processing_fct=None, gen_imgs=True):
        step = 1
        gen_loss = -1
        disc_loss = -1
        for epoch in range(epochs):
            # print("Epoch: {}, Gen_loss: {}, Disc_loss: {}, step : {}".format(epoch, gen_loss, disc_loss, step))
            start = time.time()
            b = 0  # batch nb
            for batch, labels in galaxy_generator:
                if batch_processing_fct is not None:
                    batch = batch_processing_fct(batch)
                print(batch.shape)
                # gen_loss, disc_loss = self.train_step(batch, batch_size)
                # print("Epoch: {}, Batch: {}, Step: {}, Gen_loss: {}, Disc_loss: {}".format(epoch, b, step, gen_loss, disc_loss))
                gen_loss, disc_loss = self.cgan_train_step(batch, batch_size, 1)
                print(f"gen_loss: {gen_loss}, disc_loss: {disc_loss}")
                b += 1
                if step % self.generate_every == 0:
                    display.clear_output(wait=True)
                    if gen_imgs:
                        self.generate_and_save_images(seed, "step", nb=step, vmin=-1, vmax=1)
                step += 1
                if (b >= steps_per_epoch_galaxy):
                    break

            b = 0  # batch nb
            for batch, labels in other_generator:
                if batch_processing_fct is not None:
                    batch = batch_processing_fct(batch)
                # gen_loss, disc_loss = self.train_step(batch, batch_size)
                # print("Epoch: {}, Batch: {}, Step: {}, Gen_loss: {}, Disc_loss: {}".format(epoch, b, step, gen_loss, disc_loss))
                gen_loss, disc_loss = self.cgan_train_step(batch, batch_size, 0)
                print(f"gen_loss: {gen_loss}, disc_loss: {disc_loss}")
                b += 1
                if step % self.generate_every == 0:
                    display.clear_output(wait=True)
                    if gen_imgs:
                        self.generate_and_save_images(seed, "step", nb=step, vmin=-1, vmax=1)
                step += 1
                if (b >= steps_per_epoch_other):
                    break


            if gen_imgs:
                self.generate_and_save_images(seed, "epoch", nb=epoch, vmin=-1, vmax=1)

            # Save the model every N epochs
            # NOTE: each checkpoint can be 400+ MB
            # If we checkpoint too much, it can cause serious trouble
            if (epoch + 1) % save_every == 0:
                self.model.checkpoint.save(file_prefix=self.model.checkpoint_prefix)

            print('Time for epoch {} is {} sec'.format(epoch, time.time() - start))
        # Generate after the final epoch
        display.clear_output(wait=True)
        self.generate_and_save_images(seed, "epoch", nb=epochs)

    def score(self, batch_size, epochs=10, steps_per_epoch=3):
        if(steps_per_epoch is not None):
            self.model.scorer.fit_generator(self.train_dataset_scored, steps_per_epoch=steps_per_epoch, epochs=epochs, validation_data=self.val_dataset_scored, validation_steps=100, callbacks=self.model.score_callbacks, use_multiprocessing=True, verbose=1)
        else:
            self.model.scorer.fit_generator(self.train_dataset_scored, epochs=epochs, callbacks=self.model.score_callbacks, validation_data=self.val_dataset_scored, validation_steps=100, use_multiprocessing=True, verbose=1)

    def labeling(self, labeler, X_train, y_train, X_val, y_val, batch_size=16, epochs=100, steps_per_epoch=1000, val_steps=100):
        
        labeler.labeler.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=epochs, verbose=1, steps_per_epoch=steps_per_epoch, validation_data=(X_val, y_val), validation_steps=val_steps)

    def predict(self, query_generator, query_numbers):
        predictions = self.model.scorer.predict_generator(query_generator, verbose=1)
        predictions = np.clip(predictions, a_min=0, a_max=8)
        indexed_predictions = np.concatenate([np.reshape(query_numbers, (-1, 1)), predictions], axis=1)
        print(indexed_predictions)
        np.savetxt("predictions.csv", indexed_predictions, header='Id,Predicted', delimiter=",", fmt='%d, %f', comments="")


    def generate_and_save_images(self, test_input, str, nb, vae=None, show=False, training_id="default", prime=False,
                                 vmin=0, vmax=1):
        if prime:
            mean, logvar = tf.split(self.model.generator(test_input), num_or_size_splits=2, axis=1)
            z = self.model.vae.reparameterize(mean, logvar)
            predictions = self.model.generator_prime(z, training=False)  # get generator output
        else:
            predictions = self.model.generator(test_input, training=False)  # get generator output
        #print(predictions.shape)
        if vae is not None:
            mean, logvar = tf.split(predictions, num_or_size_splits=2, axis=1)
            predictions = self.model.vae.reparameterize(mean, logvar)
            predictions = self.model.vae.decode(predictions, apply_sigmoid=True)
        fig = plt.figure(figsize=(self.fig_size, self.fig_size)) # Create a new "fig_size" inches by "fig_size" inches figure as default figure

        #print("My Predictions Are: {}".format(predictions))
        #with self.graph.as_default():
        #    set_session(self.sess)
        for i in range(predictions.shape[0]):
            image = predictions[i, :, :, 0].numpy() # take the i'th predicted image, remove the last dimension (result is 2D)
            plt.subplot(self.lines, self.cols, i+1) # consider the default figure as lines x cols grid and select the (i+1)th cell
            plt.imshow(image, cmap='gray', vmin=vmin, vmax=vmax) # plot the image on the selected cell
            plt.axis('off')
            maxval = image.max()
            minval = image.min()
        print('Max and min vals: {} {}'.format(maxval, minval))
        if show:
            plt.show() # finished plotting all images in the figure so show default figure

        if not os.path.exists(self.out_path): # create images dir if not existant
            os.mkdir(self.out_path)
        save_file = self.out_path + "/image_after_{}_{}_{}.png".format(str, nb, training_id)
        print(save_file)
        plt.savefig(save_file) # save image to dir
        return predictions

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
                    for test_batch, labels in self.train_dataset_labeled:
                        vae.random_recon_and_show(test_batch)
                        break
                    vae.sample_and_show()
        self.vprint(f"VAE trained for {epochs} epochs")

    def vprint(self, msg):
        if self.verbose:
            print(msg)
    def dprint(self, msg):
        if self.debug:
            print(msg)