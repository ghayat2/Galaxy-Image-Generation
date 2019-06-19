import tensorflow as tf
import numpy as np
import pandas as pd
import os
from PIL import Image

global_seed=5 # for reproducibilty

def preprocess_image(image):
    image = tf.image.decode_png(image, channels=1) # grayscale images => 1 channel
    image = tf.cast(image,tf.float32) / 128. - 1
    image = tf.transpose(a=image, perm=[2, 0, 1]) # images are read in channels_last format, so convert to channels_first format for performance on GPU
    return image

def load_and_preprocess_image(path):
    image = tf.read_file(path)
    image = preprocess_image(image)
    return image
    
def load_and_preprocess_image_label(path, label):
    image = tf.read_file(path)
    image = preprocess_image(image)
    label = tf.cast(label, tf.float32)
    return image, label

def read_labels2paths(data_root):
    data = pd.read_csv(os.path.join(data_root, "labeled.csv"), delimiter=',', header=0)
    
    grouped = data.groupby("Actual") # group by the label 0 or 1
    
    label2paths = {} # dict mapping each label to the list of paths to images having that label
    for label, label_data in grouped:
        ids = label_data["Id"].values
        paths = [os.path.join(data_root, "labeled", "{}.png".format(id)) for id in ids]
        label2paths[label] = paths
        
    return label2paths

def create_dataloader_train(data_root, batch_size, batches_to_prefetch=20, shuffle=True, all_data=True):
    print("Reading images paths ...")
    labels2paths = read_labels2paths(data_root)
    fake_images = labels2paths[0.0] # paths to non-galaxies
    real_images = labels2paths[1.0] # paths to galaxies
    
    if not all_data:
        print("Creating Datasets ...")
        fake_images_ds = tf.data.Dataset.from_tensor_slices(fake_images)
        real_images_ds = tf.data.Dataset.from_tensor_slices(real_images)
        
        if shuffle:
            print("Shuffling data ...")
            fake_images_ds = fake_images_ds.shuffle(buffer_size=len(fake_images), seed=global_seed)
            real_images_ds = real_images_ds.shuffle(buffer_size=len(real_images), seed=global_seed)    

        print("Mapping Data...")
        fake_images_ds = fake_images_ds.map(load_and_preprocess_image)
        real_images_ds = real_images_ds.map(load_and_preprocess_image)
        
        print("Batching Data...")
        fake_images_ds = fake_images_ds.repeat() # repeat dataset indefinitely
        fake_images_ds = fake_images_ds.batch(batch_size, drop_remainder=True).prefetch(batches_to_prefetch) # batch data (dropping remainder) and prefetch batches
        real_images_ds = real_images_ds.repeat() # repeat dataset indefinitely
        real_images_ds = real_images_ds.batch(batch_size, drop_remainder=True).prefetch(batches_to_prefetch) # batch data (dropping remainder) and prefetch batches
        
        fake_images_ds = fake_images_ds.make_one_shot_iterator().get_next() # convert to iterator
        real_images_ds = real_images_ds.make_one_shot_iterator().get_next() # convert to iterator
        
        return real_images_ds, fake_images_ds, len(real_images), len(fake_images)
    else:
        zeros = np.zeros([len(fake_images), 1])
        ones = np.ones([len(real_images), 1])
        labels = np.concatenate([zeros, ones])
        images = fake_images + real_images
        
        print("Creating Datasets ...")
        train_ds = tf.data.Dataset.from_tensor_slices((images, labels))
        
        if shuffle:
            print("Shuffling data ...")
            train_ds = train_ds.shuffle(buffer_size=len(images), seed=global_seed)
            
        print("Mapping Data...")
        train_ds = train_ds.map(load_and_preprocess_image_label)
        
        print("Batching Data...")
        train_ds = train_ds.repeat() # repeat dataset indefinitely
        train_ds = train_ds.batch(batch_size, drop_remainder=True).prefetch(batches_to_prefetch) # batch data (dropping remainder) and prefetch batches
        
        train_ds = train_ds.make_one_shot_iterator().get_next() # convert to iterator
        
        return train_ds, len(images)
    
#batch_size = 1
#real_im, fake_im, nb_reals, nb_fakes = create_dataloader_train("./data", batch_size=batch_size, batches_to_prefetch=1, all_data=False)

#print("\n\nnb_fakes:", nb_fakes,"\n\n")
#with tf.Session() as sess:
#    for i in range(nb_fakes):
#        im_vals = sess.run(fake_im)
#        
#        image = ((im_vals[0]+1)*128.0).transpose(1,2,0).astype("uint8")
#        image = Image.fromarray(image[:,:,0], mode='L') # remove the channels dimension
#        if not os.path.exists("fake_images"):
#            os.makedirs("fake_images")
#        image.save("fake_images/img_{}.png".format(i))

#batch_size = 1
#train_ds, nb_images = create_dataloader_train("./data", batch_size=batch_size, batches_to_prefetch=1, all_data=True)
#im, label = train_ds # unzip

#print("\n\nnb_images:", nb_images, "\n\n")
#with tf.Session() as sess:
#    for i in range(nb_images):
#        im_vals, label_vals = sess.run([im, label])
#        
#        lab= label_vals[0][0]
#        image = ((im_vals[0]+1)*128.0).transpose(1,2,0).astype("uint8")
#        image = Image.fromarray(image[:,:,0], mode='L') # remove the channels dimension
#        if not os.path.exists("images"):
#            os.makedirs("images")
#        image.save("images/img_{}_label_{}.png".format(i, lab))



















