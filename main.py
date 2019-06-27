from datetime import datetime
import glob
import numpy as np 
import os
from scipy import misc
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import utils
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split


from model import Model, BaseModel, CVAE, VAEGAN, TwoStepsVEAGAN, ImageRegressor, BetterD2GAN, LCGAN, DAE, MCGAN
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
    tf.compat.v1.global_variables_initializer()

    #Uncomment to create folders for labeled data
    #create_labeled_folders(data_path)

    checkpoint_dir = os.path.join(os.path.dirname( __file__ ), os.pardir, 'runs/Base_LSGAN/')
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

    ### ------------------- ###

    # Create dataset
    batch_size = 16
    data_shape = [1000, 1000, 1]
    noise_dim = 1000
    latent_dim = 100
    val_ratio = 0.1

    vae = CVAE(latent_dim)
    # vae.load_nets()
    inf_vae = tf.keras.models.clone_model(vae.inference_net)
    print(inf_vae.summary())

    # Create the labeled data generator
    #create_labeled_folders("../cosmology_aux_data_170429")

    ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
    labeled_datagen = ImageDataGenerator(preprocessing_function=utils.gan_preprocessing)
    def vae_latent(im):
        return tf.reshape(tf.squeeze(inf_vae(im)), (-1, 2*latent_dim, 1, 1))
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

    query_datagen = ImageDataGenerator()
    query_generator = query_datagen.flow_from_directory(os.path.join(data_path, "query"),
                                                        class_mode=None,
                                                        batch_size=1,
                                                        shuffle=False,
                                                        target_size=(1000,1000),
                                                        color_mode='grayscale')

    print("Data generators have been created")
    noise_dim = 63
    cgan = LCGAN(noise_dim=noise_dim, data_shape=(64, 64))

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
        [tf.keras.layers.Lambda(lambda x: x+1),
         tf.keras.layers.ZeroPadding2D(padding=(12, 12)),
         tf.keras.layers.Lambda(lambda x: x-1),
         tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
         tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
         tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
         tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
         tf.keras.layers.Reshape(target_shape=(64, 64, 1))]
    )
    """
    trainer.cgan_train(batch_size, input, galaxy_generator_64, other_generator_64,
                       epochs=1, steps_per_epoch_galaxy=1000/batch_size, steps_per_epoch_other=200/batch_size,
                       save_every=15, batch_processing_fct=resizer, gen_imgs=True)
    """
    """
    #data_path = "./cosmology_aux_data_170429/"
    model = DAE(1000, data_shape=(1000, 1000, 1), learning_rate=1e-4)
    trainer = Trainer(
        model, None, None, labeled_generator, scored_generator_train, scored_generator_val, './Results/'
    )

    trainer.dae_train(model, batch_processing_fct=None)
    """
    
    manual_feats = np.loadtxt('scoring_feats.gz')
    manual_ids = np.loadtxt('scoring_feats_ids.gz').astype(int)

    manual_dict = dict(zip(manual_ids, manual_feats))

    print(manual_feats[0])
    print(manual_ids[0])
    print(manual_dict[manual_ids[0]])

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

    all_pairs_train, all_pairs_val = train_test_split(all_pairs, test_size=0.1)

    X_train = np.array(all_pairs_train)[:, 0]
    X_val = np.array(all_pairs_val)[:, 0]


    mcgan = MCGAN(data_shape=(64, 64, 1), feat_dim=1, noise_dim=1000, checkpoint_dir=None, checkpoint_prefix=None)
    trainer = Trainer(
        mcgan, None, None, utils.custom_generator2(X_train, manual_dict), scored_generator_train, scored_generator_val, './Results/'
    )
    noise_seed = np.random.normal(0, 1, [batch_size, 1000])
    n_blobs_seed = tf.constant(np.random.randint(1, 10, [batch_size, 1]))

    trainer.mcgan_train(batch_size=batch_size, seed=tf.concat([noise_seed, n_blobs_seed], axis=-1))
    """

    mcgan = MCGAN(data_shape=(64, 64, 1), feat_dim=1, noise_dim=1000, checkpoint_dir=None, checkpoint_prefix=None)
    trainer = Trainer(
        mcgan, None, None, utils.custom_generator2(X_train, manual_dict), scored_generator_train, scored_generator_val,
        './Results/'
    )
    noise_seed = np.random.normal(0, 1, [batch_size, 1000])
    n_blobs_seed = tf.constant(np.random.randint(1, 20, [batch_size, 1]))

    trainer.mcgan_train(batch_size=batch_size, seed=tf.concat([noise_seed, n_blobs_seed], axis=-1))
    """

if __name__ == '__main__':
    main()


