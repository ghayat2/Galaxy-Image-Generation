from datetime import datetime
import glob
import numpy as np 
import os 
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import pandas as pd

from model import Model, BaseModel
from trainer import Trainer 
from dataset import Dataset, ImageLoader, ImageGen
import pathlib


def main():
    data_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), os.pardir, 'cosmology_aux_data_170429/'))
    
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

    ### Generate Labeled Data ###
    name = "labeled"

    labels_path = os.path.join(data_path, name + ".csv")
    labels = pd.read_csv(labels_path, index_col=0, skiprows=1, header=None)
    id_to_label = labels.to_dict(orient="index")
    id_to_label = {k: v[1] for k, v in id_to_label.items()}

    labeled_images_path = os.path.join(data_path, name)
    labeled_images_path = pathlib.Path(labeled_images_path)
    all_labeled = list(labeled_images_path.glob('*'))
    all_labeled = [str(p) for p in all_labeled]
    all_labels = [id_to_label[int(item.name.split('.')[0])] for item in labeled_images_path.glob('*')]
    
    ##Only get the 1 Labels###

    all_labeled = [e for e, l in zip(all_labeled, all_labels) if l == 1]
    all_ones = [1 for _ in all_labeled]

    ### ------------------- ###

    ### Generate Scored Data ###
    name = "scored"

    scores_path = os.path.join(data_path, name + ".csv")
    scores = pd.read_csv(scores_path, index_col=0, skiprows=1, header=None)
    id_to_score = scores.to_dict(orient="index")
    #id_to_score = {k: v[1] for k, v in id_to_label.items()}

    scored_images_path = os.path.join(data_path, name)
    scored_images_path = pathlib.Path(scored_images_path)
    all_scored = list(labeled_images_path.glob('*'))
    all_scored = [str(p) for p in all_scored]
    all_scores = [id_to_score[int(item.name.split('.')[0])] for item in scored_images_path.glob('*')]

    ### ------------------- ###

    # Create dataset
    batch_size = 16
    data_shape = [1000, 1000, 1]
    noise_dim = 1000

    labeled_gen = ImageGen(all_labeled, all_ones)
    scored_gen = ImageGen(all_scored, all_scores)

    # Create the model
    model = BaseModel(data_shape, noise_dim, checkpoint_dir, checkpoint_prefix)

    # Train the model
    trainer = Trainer(
        model, labeled_gen.create_dataset(), scored_gen.create_dataset().batch(batch_size), os.path.abspath(os.path.join(os.path.dirname( __file__ ), os.pardir, 'Results/'))
    )

    seed = tf.random.normal([trainer.num_examples, model.noise_dim])

    trainer.train(batch_size=batch_size, seed=seed, epochs=3)



if __name__ == '__main__':
    main()
    