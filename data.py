import tensorflow as tf
import numpy as np
import pandas as pd
import os, sys, glob
import matplotlib.pyplot as plt
from PIL import Image
import layers

global_seed=5 # for reproducibilty
ROTATIONS = False

def preprocess_image(image):
    image = tf.image.decode_png(image, channels=1) # grayscale images => 1 channel
    image = tf.cast(image,tf.float32) / 128. - 1
    if(ROTATIONS):
        rotation = tf.random.uniform([1], minval=0, maxval=3, dtype=tf.dtypes.int32)
        image = tf.image.rot90(image, k=rotation[0])
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
    
def load_and_preprocess_image_score(path, score):
    image = tf.read_file(path)
    image = preprocess_image(image)
    score = tf.cast(score, tf.float32)
    return image, score

def read_labels2paths(data_root):
    data = pd.read_csv(os.path.join(data_root, "labeled.csv"), delimiter=',', header=0)
    
    grouped = data.groupby("Actual") # group by the label 0 or 1
    
    label2paths = {} # dict mapping each label to the list of paths to images having that label
    for label, label_data in grouped:
        ids = label_data["Id"].values
        paths = [os.path.join(data_root, "labeled", "{}.png".format(id)) for id in ids]
        label2paths[label] = paths
        
    return label2paths

def create_dataloader_train_labeled(data_root, batch_size, batches_to_prefetch=20, shuffle=True, all_data=True):
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

def create_dataloader_train_scored(data_root, batch_size, batches_to_prefetch=20, shuffle=True, valid_percent=0.1):
    data = pd.read_csv(os.path.join(data_root, "scored.csv"), delimiter=',', header=0)
    data_values = data.values
    paths = [os.path.join(data_root, "scored", "{}.png".format(int(row[0]))) for row in data_values]
    scores = [row[1] for row in data_values]
    scores = np.array(scores).reshape([-1, 1])
    
    print("Creating Dataset ...")
    full_ds = tf.data.Dataset.from_tensor_slices((paths, scores))
    
    if shuffle:
        print("Shuffling data ...")
        full_ds = full_ds.shuffle(buffer_size=len(paths), seed=global_seed)
            
    print("Mapping Data...")
    full_ds = full_ds.map(load_and_preprocess_image_score)
    
    if valid_percent > 0:
        print("Creating train/validation split ...")
        nb_valid = int(valid_percent*len(paths))
        print("Validation set size: {}".format(nb_valid))
        valid_ds = full_ds.take(nb_valid)
        train_ds = full_ds.skip(nb_valid)
    else:
        nb_valid = 0
        train_ds = full_ds
        
    nb_train = len(paths) - nb_valid
    print("Train set size: {}".format(nb_train))
    
    print("Batching Data...")
    train_ds = train_ds.repeat() # repeat dataset indefinitely
    train_ds = train_ds.batch(batch_size, drop_remainder=True).prefetch(batches_to_prefetch) # batch data (dropping remainder) and prefetch batches
    train_ds = train_ds.make_one_shot_iterator().get_next() # convert to iterator
    
    if valid_percent > 0:
        valid_ds = valid_ds.repeat() # repeat dataset indefinitely
        valid_ds = valid_ds.batch(batch_size, drop_remainder=True).prefetch(batches_to_prefetch) # batch data (dropping remainder) and prefetch batches
        valid_ds = valid_ds.make_one_shot_iterator().get_next() # convert to iterator
        
    to_return = [train_ds, nb_train] # elements to return
    if valid_percent > 0:
        to_return = to_return + [valid_ds, nb_valid]
    
    print("Done.")
    return to_return

def create_dataloader_query(data_root, batches_to_prefetch=20):
    all_files = glob.glob(os.path.join(data_root, "query", "*"))

    print("Creating Dataset ...")
    full_ds = tf.data.Dataset.from_tensor_slices(all_files)
            
    print("Mapping Data...")
    full_ds = full_ds.map(load_and_preprocess_image)
    
    print("Batching Data...")
    full_ds = full_ds.repeat() # repeat dataset indefinitely
    full_ds = full_ds.batch(1, drop_remainder=True).prefetch(batches_to_prefetch) # batch data with batch_size of 1 and prefetch batches
    full_ds = full_ds.make_one_shot_iterator().get_next() # convert to iterator
    
    return full_ds, all_files, len(all_files)

def create_dataloader_train_mcgan(data_root, batch_size, batches_to_prefetch=20, shuffle=True):
    print("Reading images paths ...")
    labels2paths = read_labels2paths(data_root)
    fake_images = labels2paths[0.0] # paths to non-galaxies
    real_images = labels2paths[1.0] # paths to galaxies
    
    manual_feats = np.loadtxt(os.path.join(data_root, "features", 'labeled_feats.gz'))
    manual_ids = np.loadtxt(os.path.join(data_root, "features", 'labeled_feats_ids.gz')).astype(int)

    manual_dict = dict(zip(manual_ids, manual_feats))
    
    ids = [int(path.split("/")[-1].split(".")[0]) for path in real_images]
    feats = np.array([manual_dict[id][20] for id in ids]).reshape([-1, 1])

    print("Creating Datasets ...")
    train_ds = tf.data.Dataset.from_tensor_slices((real_images, feats))
    
    if shuffle:
        print("Shuffling data ...")
        train_ds = train_ds.shuffle(buffer_size=len(real_images), seed=global_seed)
        
    print("Mapping Data...")
    train_ds = train_ds.map(load_and_preprocess_image_label)
    
    print("Batching Data...")
    train_ds = train_ds.repeat() # repeat dataset indefinitely
    train_ds = train_ds.batch(batch_size, drop_remainder=True).prefetch(batches_to_prefetch) # batch data (dropping remainder) and prefetch batches
    
    train_ds = train_ds.make_one_shot_iterator().get_next() # convert to iterator
    
    return train_ds, len(real_images)

#batch_size = 1
#real_im, fake_im, nb_reals, nb_fakes = create_dataloader_train_labeled("./data", batch_size=batch_size, batches_to_prefetch=1, all_data=False)

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
#real_im, fake_im, nb_reals, nb_fakes = create_dataloader_train_labeled("./data", batch_size=batch_size, batches_to_prefetch=1, all_data=False)

#real_im = layers.max_pool_layer(real_im, pool_size=(2,2), strides=(2,2), padding=(12,12))
#real_im = layers.max_pool_layer(real_im, pool_size=(2,2), strides=(2,2))
#real_im = layers.max_pool_layer(real_im, pool_size=(2,2), strides=(2,2))
#real_im = layers.max_pool_layer(real_im, pool_size=(2,2), strides=(2,2))

#SAVE_DIR = "./images/real_images_MAX_POOL_1"
#print("\n\nnb_reals:", nb_reals,"\n\n")
#with tf.Session() as sess:
#    for i in range(nb_reals):
#        im_vals = sess.run(real_im)
#        fig = plt.figure(figsize=(10, 10))

#        image = ((im_vals[0]+1)*128.0).transpose(1,2,0).astype("uint8")[:,:,0]
#        plt.subplot(1, 1, 1)
#        min_val = image.min()
#        max_val = image.max()
#        plt.imshow(image, cmap='gray', vmin=0, vmax=255) # plot the image on the selected cell
#        plt.axis('off')
#        plt.title("min: {}, max: {}".format(min_val, max_val))
#        
#        if not os.path.exists(SAVE_DIR):
#            os.makedirs(SAVE_DIR)
#        fig.savefig(os.path.join(SAVE_DIR, "img_{}.png".format(i))) # save image to dir
#        plt.close()

#batch_size = 1
#train_ds, nb_images = create_dataloader_train_labeled("./data", batch_size=batch_size, batches_to_prefetch=1, all_data=True)
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

#batch_size = 16
#train_ds, nb_train, valid_ds, nb_valid = create_dataloader_train_scored("./data", batch_size=batch_size, batches_to_prefetch=1, valid_percent=0.1)
#im_train, score_train = train_ds # unzip
#im_valid, score_valid = valid_ds # unzip

#print(im_train.shape)
#print(score_train.shape)

#print(create_dataloader_query(data_root="./data", batches_to_prefetch=20))


#print(create_dataloader_train_mcgan(data_root="./data", batch_size=16, batches_to_prefetch=2, shuffle=True))











