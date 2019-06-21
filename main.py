from datetime import datetime
import glob
import numpy as np 
import os 
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.backend import set_session
import pandas as pd
import utils
import matplotlib.pyplot as plt

import sklearn
from sklearn.model_selection import train_test_split

from model import Model, BaseModel, CVAE, VAEGAN, TwoStepsVEAGAN
from trainer import Trainer 
from dataset import Dataset, ImageLoader, ImageGen
import pathlib, time

"""
    The REGRESSOR_TYPE Flag specifies the type of regressor that will be
        trained on the VAE latent space and will be used to make predictions
        on the scored image dataset
        Options: None (default: Use different model to output prediction), 
                 Random Forest, Ridge, MLP
"""
REGRESSOR_TYPE = None  #Random Forest, Ridge, MLP

#Function to try to ignore the dataset class and just have a Keras DataGenerator pipeline
def flow_from_dataframe(img_data_gen, in_df, path_col, y_col, batch_size=16, subset='training', **dflow_args):
    base_dir = os.path.dirname(in_df[path_col].values[0])
    print('## Ignore next message from keras, values are replaced anyways')
    df_gen = img_data_gen.flow_from_directory(base_dir, 
                                     class_mode = 'sparse',
                                     batch_size=batch_size,
                                     target_size=(1000, 1000),
                                     subset=subset,
                                     color_mode='grayscale',
                                    **dflow_args)
    df_gen.filenames = in_df[path_col].values
    df_gen._filepaths = df_gen.filenames
    df_gen.classes = np.stack(in_df[y_col].values)
    df_gen.samples = in_df.shape[0]
    df_gen.n = in_df.shape[0]
    df_gen.directory = '' # since we have the full path
    df_gen._set_index_array()           
    print('Reinserting dataframe: {} images'.format(in_df.shape[0]))
    return df_gen

def create_labeled_folders(data_path):

    #Create labeled dirs:
    name = "labeled"
    labels_path = os.path.join(data_path, name + ".csv")
    labels = pd.read_csv(labels_path, index_col=0, skiprows=1, header=None)
    id_to_label = labels.to_dict(orient="index")
    id_to_label = {k: v[1] for k, v in id_to_label.items()}

    labeled_images_path = os.path.join(data_path, name)
    labeled_images_path = pathlib.Path(labeled_images_path)
    onlyFiles = [f for f in os.listdir(labeled_images_path) if (os.path.isfile(os.path.join(labeled_images_path, f)) and (f != None))]
    all_indexes = [item.split('.')[0] for item in onlyFiles]
    all_indexes = filter(None, all_indexes)
    all_pairs = [[item, id_to_label[int(item)]] for item in all_indexes]

    # Add if does not exist
    if(~os.path.isdir(os.path.join(data_path, name, '0'))):
        os.mkdir(os.path.join(data_path, name, '0'))
    if(~os.path.isdir(os.path.join(data_path, name, '1'))):
        os.mkdir(os.path.join(data_path, name, '1'))

    for file, label in all_pairs:
        print(os.path.join(data_path, file))
        print(os.path.join(data_path, "{}".format(int(label)), file))
        os.rename(os.path.join(labeled_images_path, file + '.png'), os.path.join(labeled_images_path, "{}".format(int(label)), file + '.png')) 


def main():
        
    print(tf.executing_eagerly())
    data_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), os.pardir, 'cosmology_aux_data_170429/'))
    sess = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()
    set_session(sess)
    tf.compat.v1.global_variables_initializer()

    #Uncomment to create folders for labeled data
    #create_labeled_folders(data_path)

    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

    ### ------------------- ###

    # Create dataset
    batch_size = 16
    data_shape = [1000, 1000, 1]
    noise_dim = 1000
    latent_dim = 100
    val_ratio = 0.1

    vae = CVAE(latent_dim)
    vae.load_nets()
    inf_vae = tf.keras.models.clone_model(vae.inference_net)
    print(inf_vae.summary())

    # Create the labeled data generator
    #create_labeled_folders("../cosmology_aux_data_170429")
    def vae_latent(im):
        return tf.reshape(tf.squeeze(inf_vae(im)), (-1, 2*latent_dim, 1, 1))

    labeled_datagen = ImageDataGenerator(preprocessing_function=utils.vae_preprocessing)
    labeled_generator = labeled_datagen.flow_from_directory(os.path.join(data_path, "labeled"), 
                                        class_mode='binary', 
                                        batch_size=batch_size, 
                                        target_size=(1000, 1000),
                                        color_mode='grayscale')

    
    #Added Validation Split value here, but not sure if it is compatible
    #with the custom flow_from_dataframe function above

    scored_datagen_train = ImageDataGenerator(preprocessing_function=utils.vae_preprocessing, validation_split=0.5)
    scored_datagen_val = ImageDataGenerator(preprocessing_function=utils.vae_preprocessing, validation_split=0.5)
    scores_path = os.path.join(data_path, "scored.csv")
    scores = pd.read_csv(scores_path, index_col=0, skiprows=1, header=None)
    id_to_score = scores.to_dict(orient="index")
    id_to_score = {k: v[1] for k, v in id_to_score.items()}

    scored_images_path = os.path.join(data_path, "scored")
    scored_images_path = pathlib.Path(scored_images_path)
    onlyFiles = [f for f in os.listdir(scored_images_path) if (os.path.isfile(os.path.join(scored_images_path, f)) and (f != None))]

    all_indexes = [item.split('.')[0] for item in onlyFiles]
    all_indexes = filter(None, all_indexes)
    all_pairs = [[os.path.join(scored_images_path, item) + '.png', id_to_score[int(item)]] for item in all_indexes]

    all_pairs_train, all_pairs_val = train_test_split(all_pairs, test_size=val_ratio)

    scored_df_train = pd.DataFrame(all_pairs_train, columns=['Path', 'Value'])
    scored_df_val = pd.DataFrame(all_pairs_val, columns=['Path', 'Value'])

    scored_generator_train = flow_from_dataframe(scored_datagen_train, scored_df_train, 'Path', 'Value', batch_size=batch_size, subset='training')
    scored_generator_val = flow_from_dataframe(scored_datagen_val, scored_df_val, 'Path', 'Value', batch_size=batch_size, subset='validation')

    query_images_path = os.path.join(data_path, "query")
    query_images_path = pathlib.Path(query_images_path)
    only_files = [f for f in os.listdir(query_images_path) if (os.path.isfile(os.path.join(query_images_path, f)) and (f != None))]
    all_indexes = [int(item.split('.')[0]) for item in only_files]
    sorted_queries = np.sort(all_indexes)

    print(np.reshape(sorted_queries, (-1, 1)))

    query_datagen = ImageDataGenerator()
    query_generator = query_datagen.flow_from_directory(os.path.join(data_path, "query"),
                                                        class_mode=None,
                                                        batch_size=1,
                                                        shuffle=False,
                                                        target_size=(1000,1000),
                                                        color_mode='grayscale')


    print("Data generators have been created")


    # -------------------- #

    if REGRESSOR_TYPE:
        utils.predict_with_regressor(vae, REGRESSOR_TYPE, scored_generator_train, query_generator, sorted_queries, epochs=100)
        return

    # # Create the model
    model = TwoStepsVEAGAN(vae, data_shape, 2*latent_dim, checkpoint_dir, checkpoint_prefix, reload_ckpt=False)

    # Train the model
    trainer = Trainer(
        model, sess, graph, labeled_generator, scored_generator_train, scored_generator_val, os.path.abspath(os.path.join(os.path.dirname( __file__ ), os.pardir, 'Results/'))
    )

    seed = np.random.normal(0, 1, [trainer.num_examples, model.noise_dim])

    #Specify epochs, steps_per_epoch, save_every
    trainer.vaegan_train(batch_size=batch_size, seed=seed, epochs=1,
                  steps_per_epoch=3,
                  batch_processing_fct=vae_latent,
                  gen_imgs=True)
    print("Generator trained")

    trainer.two_steps_vaegan_train(batch_size=batch_size, seed=seed, epochs=1,
                         steps_per_epoch=3,
                         batch_processing_fct=None,
                         gen_imgs=True)


if __name__ == '__main__':
    main()


