from datetime import datetime
import glob
import numpy as np 
import os 
from PIL import Image
import skimage
import tensorflow as tf
from tensorflow import keras 

from model import Model, BaseModel
from trainer import Trainer 
from dataset import Dataset
import tensorflow.contrib.eager as tfe


def main():

    data_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), os.pardir, 'Semester 4/cosmology_aux_data_170429/'))
    labeled_1_dataset = Dataset(data_path, "test", hasLabels=True, wantLabel=1)

    # Create dataset
    batch_size = 16
    data_shape = [1000, 1000, 1]
    noise_dim = 1000

    labeled_1_dataset.create_tf_dataset(batch_size=16)

    # Create the model
    model = BaseModel(data_shape, noise_dim)

    # Train the model
    trainer = Trainer(
        model, labeled_1_dataset, os.path.abspath(os.path.join(os.path.dirname( __file__ ), os.pardir, 'Results/'))
    )

    seed = tf.random.normal([trainer.num_examples, model.noise_dim])

    trainer.train(seed=seed, epochs=1)



if __name__ == '__main__':
    main()
    