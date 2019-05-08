import glob
import numpy as np 
import os
import skimage
import tensorflow as tf 
import pandas as pd
import pathlib



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

        self.dataset_tf = self.dataset_tf.shuffle(3)
        self.dataset_tf = self.dataset_tf.batch(self.batch_size)

        print("My Batches: {}".format(self.dataset_tf))
