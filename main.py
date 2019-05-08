import tensorflow as tf
import os, sys
import time
import numpy as np
import matplotlib.pyplot as plt
import PIL
import cv2
import pathlib
import pandas as pd
import time

from tensorflow.keras import layers
from IPython import display

tf.enable_eager_execution()
print("\n" + tf.__version__ + "\n")

# ---------------------------------------------------Constants--------------------------------------------------------
EPOCHS = 500
BATCH_SIZE = 16
train_fig_size = 20 # in inches
lines = cols = 2
num_examples_to_generate = lines*cols # number of generated images equal to number of real images provided 
noise_dim = 1500
generate_every = 20 # nb of steps between every generation

train_out_images_dir = "./train_out_images/"
test_out_images_dir = "./test_out_images/"
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
continue_training = False
test_image_nb = 0

# -------------------------------------------------Read data paths----------------------------------------------------

data_path = os.path.join(os.getcwd(), "data")
labels_path = os.path.join(data_path, "labeled.csv")
labels = pd.read_csv(labels_path, index_col=0, skiprows=1, header=None)
id_to_label = labels.to_dict(orient="index")
id_to_label = {k:v[1] for k,v in id_to_label.items()}

labeled_images_path = os.path.join(data_path, "labeled")
labeled_images_path = pathlib.Path(labeled_images_path)
all_labeled = list(labeled_images_path.glob('*'))
all_labeled = [str(p) for p in all_labeled]
all_labels = [id_to_label[int(item.name.split('.')[0])] for item in labeled_images_path.glob('*')]
print(len(all_labeled))

scored_images_path = os.path.join(data_path, "scored")
scored_images_path = pathlib.Path(scored_images_path)
all_scored = list(scored_images_path.glob('*'))
all_scored = [str(p) for p in all_labeled]
# Make dataset only with images labeled as galaxies
all_labeled = [e for e, l in zip(all_labeled, all_labels) if l]
data_size = len(all_labeled)
print(data_size)

# ---------------------------------------Image loading/saving and Preprocessing---------------------------------------

def preprocess_image(image):
    image = tf.image.convert_image_dtype(tf.image.decode_png(image, channels=1), tf.float32)
    image = (image - 0.5) / 0.5 # normalize to [-1, 1] range 
      
    return image

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)

def generate_and_save_images(model, test_input, fig_size, lines, cols, dir, str, nb):
    predictions = model(test_input, training=False) # get generator output

    fig = plt.figure(figsize=(fig_size, fig_size)) # Create a new "fig_size" inches by "fig_size" inches figure as default figure

    for i in range(predictions.shape[0]):
        image = predictions[i, :, :, 0] # take the i'th predicted image, remove the last dimension (result is 2D)
        plt.subplot(lines, cols, i+1) # consider the default figure as lines x cols grid and select the (i+1)th cell
        plt.imshow(image, cmap='gray', vmin=-1.0, vmax=1.0) # plot the image on the selected cell
        plt.axis('off')
        maxval = image.numpy().max()
        minval = image.numpy().min()
        print('Max and min vals: {} {}'.format(maxval, minval))
#    plt.show() # finished plotting all images in the figure so show default figure

    if not os.path.exists(dir): # create images dir if not existant
        os.mkdir(dir)
    plt.savefig(dir + "image_after_{}_{}.png".format(str, nb)) # save image to dir
    return predictions

# --------------------------------------------------Models-----------------------------------------------------------
def make_generator_model(noise_dim=1000):
    model = tf.keras.Sequential()

    model.add(layers.Dense(5*5*256, use_bias=False, input_shape=(noise_dim,)))
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

def make_discriminator_model():
    model = tf.keras.Sequential()
  
    model.add(layers.Conv2D(4, (3, 3), strides=(2, 2), padding='same',
                                     input_shape=[1000, 1000, 1]))
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

# ---------------------------------------------Losses and Optimizers---------------------------------------------------
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)
  
generator_optimizer = tf.train.AdamOptimizer(1e-3)
discriminator_optimizer = tf.train.AdamOptimizer(1e-4)

# ------------------------------------------------Train Loop-----------------------------------------------------------
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim]) # number of generated images equal to number of real images provided 
                                                      # to discriminator (i,e batch_size)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return gen_loss, disc_loss

def train(dataset, epochs, seed):
    step = 0
    for epoch in range(epochs):
        start = time.time()

        b = 0 # batch nb
        for image_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch)
            print("Epoch: {}, Batch: {}, Step: {}, Gen_loss: {}, Disc_loss: {}".format(epoch, b, step, gen_loss, disc_loss))
            b += 1
            if step % generate_every == 0:
                display.clear_output(wait=True)
                generate_and_save_images(generator, seed, train_fig_size, lines, cols, train_out_images_dir, "step", nb = step)
            step +=1

        display.clear_output(wait=True)
        generate_and_save_images(generator, seed, train_fig_size, lines, cols, train_out_images_dir, "epoch", nb = epoch)

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print ('Time for epoch {} is {} sec'.format(epoch, time.time()-start))

    # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(generator, seed, train_fig_size, lines, cols, train_out_images_dir, "epoch", nb = EPOCHS)

# --------------------------------------------------------------------------------------------------------------------

# -----------------------------------------------------Start----------------------------------------------------------
# Generator
generator = make_generator_model(noise_dim)

# Discriminator
discriminator = make_discriminator_model()

#Checkpoints
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# Restore from lastest available checkpoint
latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir) # returns None if no checkpoint found
checkpoint_found = (latest_checkpoint is not None) # model can be restored if a checkpoint found

if checkpoint_found:
    status = checkpoint.restore(latest_checkpoint)

if checkpoint_found and not continue_training: # test restored model
    noise = tf.random.normal([1, noise_dim])
    generate_and_save_images(generator, noise, 12, 1, 1, test_out_images_dir, "restore", nb = test_image_nb)
    sys.exit(0)

# Pretest generator and discriminator
seed = tf.random.normal([num_examples_to_generate, noise_dim])
generated_images = generate_and_save_images(generator, seed, train_fig_size, lines, cols, train_out_images_dir, "pretrain", nb = 0) # test generator output on "num_examples_to_generate" noise images (before training)
decision = discriminator(generated_images) # test discriminator on the generated images
print(decision)

# Read Training Data
AUTOTUNE = tf.data.experimental.AUTOTUNE # constant to autotune the number of parallel calls when reading dataset

path_ds = tf.data.Dataset.from_tensor_slices(all_labeled)
train_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE)

# Train
train(train_ds, EPOCHS, seed)

# read test dataset
# path_ds = tf.data.Dataset.from_tensor_slices(all_scored)
# test_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
