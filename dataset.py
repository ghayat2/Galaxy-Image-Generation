import glob
import numpy as np 
import os
import skimage
import tensorflow as tf 
import pandas as pd
import pathlib
from sklearn.model_selection import train_test_split


class Dataset:

    def __init__(self, data_path, name, hasLabels=False, wantLabel=None):

        self.name = name

        self.num_batches = None

        self.hasLabels = hasLabels
        self.wantLabel = wantLabel

        self.data_path = data_path
        self.read_data()

        self.num_data = self.data.shape[0]


    def preprocess(self, img):

        img /= 255.0
        img = (img - 0.5) / 0.5 # normalize to [-1, 1] range

        return img

    def read_data(self):
        # Get list of files

        data = []

        if (self.hasLabels):

            labels_path = os.path.join(self.data_path, self.name + ".csv")
            labels = pd.read_csv(labels_path, index_col=0, skiprows=1, header=None)
            id_to_label = labels.to_dict(orient="index")
            id_to_label = {k:v[1] for k,v in id_to_label.items()}

            labeled_images_path = os.path.join(self.data_path, self.name)
            labeled_images_path = pathlib.Path(labeled_images_path)
            all_labeled = list(labeled_images_path.glob('*'))
            all_labeled = [str(p) for p in all_labeled]
            all_labels = [id_to_label[int(item.name.split('.')[0])] for item in labeled_images_path.glob('*')]

            if(self.wantLabel != None):
                all_labeled = [e for e, l in zip(all_labeled, all_labels) if l == self.wantLabel]

            for img_path in all_labeled:
                img_name = os.path.basename(img_path)
                
                img = skimage.io.imread(img_path).astype(np.float32)
                
                preprocessed = self.preprocess(img)

                # Append to output list
                data.append(preprocessed.astype(np.float32))

            self.data = np.stack(data, axis=0)

            self.mean = self.data.mean()
            self.std  = self.data.std()

        else:
            print("Has No Labels!")


    def create_tf_dataset(self, batch_size=16):
        self.batch_size = batch_size
        self.num_batches = np.ceil(self.num_data / self.batch_size).astype(int)

        self.dataset_tf = tf.data.Dataset.from_tensor_slices(self.data)

        self.dataset_tf = self.dataset_tf.shuffle(20)
        self.dataset_tf = self.dataset_tf.batch(self.batch_size)

        print("My Batches: {}".format(self.dataset_tf))

class ImageGen():
    """
    Generator for image datasets that do not fit in memory
    """

    def __init__(self, all_paths, all_labels, img_loader=None):
        if img_loader is None:
            img_loader=ImageLoader()
        self.img_loader = img_loader
        self.all_paths = all_paths
        self.all_labels = all_labels
        self.curr = 0
        self.len = len(all_labels)
        print(self.len)

    def __len__(self):
        return self.len

    def get_next(self):
        for x, y in zip(self.all_paths, self.all_labels):
            self.curr += 1
            #self.curr %= self.len
            if self.curr%100==0:
                print(self.curr)
            if self.curr == self.len:
                self.curr = 0
            yield (x, self.curr)

    def create_dataset(self, batch_size=16):
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        path_ds = tf.data.Dataset.from_generator(self.get_next, (tf.string, tf.int32))
        scored_ds = path_ds.map(self.img_loader.load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
        return scored_ds


class ImageLoader:
    def __init__(self):
        pass

    def preprocess_image(self, image):
        image = tf.image.convert_image_dtype(tf.image.decode_png(image, channels=1), tf.float32)
        image = (image - 0.5) / 0.5
        return image


    def load_and_preprocess_image(self, path, label):
        image = tf.io.read_file(path)
        return self.preprocess_image(image), label

class VAELoader(ImageLoader):
    def preprocess_image(self, image):
        image = tf.image.convert_image_dtype(tf.image.decode_png(image, channels=1), tf.float32)
        return image