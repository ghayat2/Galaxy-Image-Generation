from datetime import datetime
import glob
import numpy as np 
import os 
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import utils
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
import sys

from model import Model, BaseModel, CVAE, VAEGAN, TwoStepsVEAGAN, ImageRegressor, BetterD2GAN, CGAN
from trainer import Trainer 
from dataset import Dataset, ImageLoader, ImageGen
import pathlib, time



def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()    
    keys_list = [keys for keys in flags_dict]    
    for keys in keys_list:
        FLAGS.__delattr__(keys)

del_all_flags(tf.flags.FLAGS)

## REGRESSOR PROPERTIES ##
tf.flags.DEFINE_string("regressor_type", None, """specifies the type of regressor that will betrained on the VAE latent space 
                      and will be used to make predictionson the query image dataset
                      Options: None (default: Use different model to output predictions), Random Forest, Ridge, MLP""")
tf.flags.DEFINE_bool("vae_encoded_images", False, "True if the images of scored and query dataset were previously encoded by the vae model")
tf.flags.DEFINE_bool("only_features", False, "Train the regressor only on manually crafted features")
tf.flags.DEFINE_integer("feature_dim", 34, "Number of manually crafted features")
tf.flags.DEFINE_integer("latent_dim", 100, "The dimension of the latent space of the vae")

FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)

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
    """Creates two folders within the labeled folder of the cosmology_aux_data_170429 folder,
    separating images associated to a label of 0 and of 1
    :param str data_path: Containing the path of the cosmology_aux_data_170429 folder
    """
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
    """ The main method of our project takes care of running our model and/or 
    predicting scores of the query images
    """
    
    print("Eager execution (required): ", tf.executing_eagerly())
    data_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), os.pardir, 'cosmology_aux_data_170429/'))
    tf.compat.v1.global_variables_initializer()
    #Uncomment to create folders for labeled data
    #create_labeled_folders(data_path)

    ### ------------------- ###

    # Create dataset
    batch_size = 16
    latent_dim = 100
    val_ratio = 0.1

    vae = CVAE(latent_dim)
    vae.load_nets()
    inf_vae = tf.keras.models.clone_model(vae.inference_net)
    print(inf_vae.summary())

    ### Creating the generators to yield images in the labeled folder ###
    ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
    labeled_datagen = ImageDataGenerator(preprocessing_function=utils.gan_preprocessing)
    labeled_generator = labeled_datagen.flow_from_directory(os.path.join(data_path, "labeled"), 
                                        class_mode='binary',
                                        classes=['1'], 
                                        batch_size=batch_size, 
                                        target_size=(1000, 1000),
                                        color_mode='grayscale')
    
    labeled_datagen_gan = ImageDataGenerator(preprocessing_function=utils.gan_preprocessing)
    labeled_generator_gan = labeled_datagen.flow_from_directory(os.path.join(data_path, "labeled"),
                                                            class_mode='binary',
                                                            batch_size=batch_size,
                                                            target_size=(1000, 1000),
                                                            color_mode='grayscale')
    def vae_latent(im):
        return tf.reshape(tf.squeeze(inf_vae(im)), (-1, 2*latent_dim, 1, 1))



    ## Preprocessing data to create generators interating through the scored images dataset ###
    
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

    ## Preprocessing data to create generators interating through the query image dataset ###
    
    query_images_path = os.path.join(data_path, "query/1")
    query_images_path = pathlib.Path(query_images_path)
    only_files = [f for f in os.listdir(query_images_path) if (os.path.isfile(os.path.join(query_images_path, f)) and (f != None))]
    all_indexes = [int(item.split('.')[0]) for item in only_files]
    sorted_queries = np.sort(all_indexes)

    query_datagen = ImageDataGenerator()
    query_generator = query_datagen.flow_from_directory(os.path.join(data_path, "query/1"),
                                                        class_mode=None,
                                                        batch_size=1,
                                                        shuffle=False,
                                                        target_size=(1000,1000),
                                                        color_mode='grayscale')
    
    

    ### Preprocessing data to create generators interating through the scored        ###
    ### and query dataset and associate features to the images. Note that the        ###
    ### image features need to be included in the folder given by the following path ###
    
    feature_path = os.path.join(data_path, "features/")
    manual_score_feats = np.loadtxt(feature_path + 'scoring_feats.csv')
    manual_score_ids = np.loadtxt(feature_path + 'scoring_feats_ids.csv').astype(int)
    manual_query_feats = np.loadtxt(feature_path + 'query_feats.csv')
    manual_query_ids = np.loadtxt(feature_path + 'query_feats_ids.csv').astype(int)

    print("Shape manual score features", manual_score_feats.shape)
    print("Shape manual score ids", manual_score_ids.shape)
    print("Shape manual query features", manual_query_feats.shape)
    print("Shape manual query ids", manual_query_ids.shape)


    manual_score_dict = dict(zip(manual_score_ids, manual_score_feats))
    manual_query_dict = dict(zip(manual_query_ids, manual_query_feats))

    image_score_list = [os.path.join(scored_images_path, str(i) + ".png") for i in manual_score_ids]
    scored_feature_generator = utils.custom_generator(image_score_list, manual_score_dict)

    image_query_list = [os.path.join(query_images_path, str(i) + ".png") for i in manual_query_ids]
    query_feature_generator = utils.custom_generator(image_query_list, manual_query_dict, score= False, batch_size=1)

    print("Data generators have been created")
    
    if FLAGS.regressor_type:
        utils.predict_with_regressor(vae, FLAGS.regressor_type, scored_feature_generator, 
                                     query_feature_generator, sorted_queries, vae_encoded_images=FLAGS.vae_encoded_images, 
                                     only_features=FLAGS.only_features, feature_dim=FLAGS.feature_dim, latent_dim=FLAGS.latent_dim)
        return
    
    noise_dim = 63
    cgan = CGAN(noise_dim=noise_dim, data_shape=(64, 64))

    galaxy_generator_64 = labeled_datagen.flow_from_directory(os.path.join(data_path, "labeled"),
                                                            class_mode='binary',
                                                            classes=['1'],
                                                            batch_size=batch_size,
                                                            target_size=(64, 64),
                                                            color_mode='grayscale')

    other_generator_64 = labeled_datagen.flow_from_directory(os.path.join(data_path, "labeled"),
                                                              class_mode='binary',
                                                              classes=['0'],
                                                              batch_size=batch_size,
                                                              target_size=(64, 64),
                                                              color_mode='grayscale')
    noise = np.random.normal(0, 1, [batch_size, noise_dim])
    input = tf.concat([noise, tf.constant(1.0, shape=(batch_size, 1))], axis=-1)
    trainer = Trainer(
        cgan, None, None, labeled_generator, scored_generator_train, scored_generator_val, './Results/'
    )

    g = cgan.generator(input)
    print(g.shape)

    resizer = tf.keras.Sequential(
        [tf.keras.layers.ZeroPadding2D(padding=(12, 12)),
         tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
         tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
         tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
         tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))]
    )

    trainer.cgan_train(batch_size, input, galaxy_generator_64, other_generator_64,
                       epochs=1, steps_per_epoch_galaxy=1000/batch_size, steps_per_epoch_other=200/batch_size,
                       save_every=15, batch_processing_fct=resizer, gen_imgs=True)

if __name__ == '__main__':
    main()


