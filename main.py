import glob
import numpy as np 
import os, sys
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import utils
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from datetime import datetime
import time
from argparse import ArgumentParser

tf.enable_eager_execution() # start eager mode

from model import *
from trainer import Trainer 
from dataset import Dataset, ImageLoader, ImageGen
import pathlib

parser = ArgumentParser()
parser.add_argument('-rt', '--regressor_type', type = str, default = None, choices=["Random_Forest", "Ridge", "MLP", "Boost"], 
                        help = 'type of regressor trained on the VAE latent space and used to make predictionson the query image dataset')
parser.add_argument('-only_f', '--only_features', help = 'whether to train the regressor only on manually crafted features', action="store_true")   
parser.add_argument('-f_dim', '--feature_dim', type = int, default = 38, help = 'Number of manually crafted features')
parser.add_argument('-lat_dim', '--latent_dim', type = int, default = 100, help = 'The dimension of the latent space of the vae')
parser.add_argument('-bs', '--batch_size', type = int, default = 16, help = 'size of training batch for VAE')
parser.add_argument('-vr', '--val_ratio', type = float, default = 0.1, help = 'percentage of the data to use for validation')

args = parser.parse_args()

REGRESSOR_TYPE=args.regressor_type
ONLY_FEATURES=args.only_features
FEATURES_DIM=args.feature_dim
LATENT_DIM=args.latent_dim
BATCH_SIZE=args.batch_size
VAL_RATIO=args.val_ratio
H, W = 1000, 1000

DATA_ROOT = "./data"

print("\n")
print("Run infos:")
print("    REGRESSOR_TYPE: {}".format(REGRESSOR_TYPE))
print("    ONLY_FEATURES: {}".format(ONLY_FEATURES))
print("    FEATURES_DIM: {}".format(FEATURES_DIM))
print("    LATENT_DIM: {}".format(LATENT_DIM))
print("    BATCH_SIZE: {}".format(BATCH_SIZE))
print("    VAL_RATIO: {}".format(VAL_RATIO))
print("\n")
sys.stdout.flush()

#sys.exit(0)

def timestamp():
    return datetime.fromtimestamp(time.time()).strftime("%Y.%m.%d-%H:%M:%S")

def main():
    """ The method that takes care of running the model and/or 
    predicting scores of the query images
    """
    
    # Create dataset
    print("Start at {} ...".format(timestamp()))
    utils.create_labeled_folders(data_root=DATA_ROOT)
    print("Eager execution (required): ", tf.executing_eagerly(), "\n")
    tf.compat.v1.global_variables_initializer()
#    sys.exit(0)

    vae = CVAE(LATENT_DIM)
    if not ONLY_FEATURES: # skip training VAE if using only manually extracted features for training score regressor
    
        # -----TODO: add code for training VAE before loading checkpoint ----------
        
        vae.load_nets()
        inf_vae = tf.keras.models.clone_model(vae.inference_net)
        print(inf_vae.summary())

    ## Preprocessing data to create generators iterating through the query image dataset ###
    
    query_images_path = os.path.join(DATA_ROOT, "query")
    query_images_path = pathlib.Path(query_images_path)
    only_files = [f for f in os.listdir(query_images_path) if (os.path.isfile(os.path.join(query_images_path, f)) and (f != None))]
    all_indexes = [int(item.split('.')[0]) for item in only_files]
    sorted_queries = np.sort(all_indexes)

    ### Preprocessing data to create generators iterating through the scored        ###
    ### and query dataset and associate features to the images. Note that the        ###
    ### image features need to be included in the folder given by the following path ###
    
    feature_path = os.path.join(DATA_ROOT, "features/")
    manual_score_feats = np.loadtxt(feature_path + 'scored_feats.gz')
    manual_score_ids = np.loadtxt(feature_path + 'scored_feats_ids.gz').astype(int)
    manual_query_feats = np.loadtxt(feature_path + 'query_feats.gz')
    manual_query_ids = np.loadtxt(feature_path + 'query_feats_ids.gz').astype(int)

    print("\nShape manual score features", manual_score_feats.shape)
    print("Shape manual score ids", manual_score_ids.shape)
    print("Shape manual query features", manual_query_feats.shape)
    print("Shape manual query ids", manual_query_ids.shape)


    manual_score_dict = dict(zip(manual_score_ids, manual_score_feats))
    manual_query_dict = dict(zip(manual_query_ids, manual_query_feats))

    image_score_list = [os.path.join(DATA_ROOT, "scored", str(i) + ".png") for i in manual_score_ids]
    scored_feature_generator = utils.custom_generator(image_score_list, manual_score_dict, FEATURES_DIM)

    image_query_list = [os.path.join(DATA_ROOT, "query", str(i) + ".png") for i in manual_query_ids]
    query_feature_generator = utils.custom_generator(image_query_list, manual_query_dict, FEATURES_DIM, do_score= False, batch_size=1)

    print("Data generators have been created")
    
#    sys.exit(0)
    
    if REGRESSOR_TYPE is not None:
        utils.predict_with_regressor(vae, REGRESSOR_TYPE, scored_feature_generator, 
                                     query_feature_generator, sorted_queries, 
                                     only_features=ONLY_FEATURES, feature_dim=FEATURES_DIM, latent_dim=LATENT_DIM, save_root=os.path.join("./Regressor", REGRESSOR_TYPE))
    
    print("End at {} ...".format(timestamp()))
    
if __name__ == '__main__':
    main()


