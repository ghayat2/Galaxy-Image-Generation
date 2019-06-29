import random
import os, sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model, neural_network
import tensorflow as tf
import time as t
import random
import pathlib
from shutil import copyfile
from tqdm import tqdm
import xgboost as xgb

from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense
from keras.preprocessing import image as krs_image

from sklearn import model_selection
from sklearn.externals import joblib
from sklearn.datasets import load_sample_image
from skimage.feature import blob_doh, blob_log
from skimage.exposure import histogram
from skimage.feature import shape_index
from skimage.measure import shannon_entropy
from sklearn import neural_network


# ------------------------------------------------------------------ CUSTOM DATA GENERATORS --------------------------------------------------------------------------

def custom_generator(images_list, manual_dict, do_score=True, batch_size=16):
    """ Returns an numpy array of size batch_size containing the decoded images,
    the features associated with these images and their scores.
    
    :param array image_list: An array containing the full path of the images to be sorted below
    :param dict manual_dict: A dictionary associating the id of an image to 
    its features ([:34]) and its score ([34])
    :param bool score: Wether the images are scored or not
    :param int batch_size: The batch_size to use
    :rtype: Numpy array
    """
    i = 0
    images_list = np.sort(images_list) # sort image paths
    
    while i<len(images_list):
        batch = {'images': [], 'manual': [], 'scores': []}
        for b in range(batch_size):
            # Read image from list and convert to array
            image_path = images_list[i]
            image_name = int(os.path.basename(image_path).replace('.png', ''))
            image = krs_image.load_img(image_path, target_size=(1000, 1000), color_mode='grayscale')
            image = gan_preprocessing(krs_image.img_to_array(image))

            manual_features = manual_dict[image_name][:34]

            score = manual_dict[image_name][34] if do_score else 0

            batch['images'].append(image)
            batch['manual'].append(manual_features)
            batch['scores'].append(score)

            i += 1

        batch['images'] = np.array(batch['images'])
        batch['manual'] = np.array(batch['manual'])
        batch['scores'] = np.array(batch['scores'])

        yield [batch['images'], batch['manual']], batch['scores']

def custom_generator2(images_list, manual_dict, batch_size=16):
    """
    Generator that yields the images along with their hand-crafted features
    """
    i = 0
    while True:
        batch = {'images': [], 'manual': []}
        for b in range(batch_size):
            if i == len(images_list):
                i = 0
                random.shuffle(images_list)
            # Read image from list and convert to array
            image_path = images_list[i]
            image_name = int(os.path.basename(image_path).replace('.png', ''))
            image = krs_image.load_img(image_path, target_size=(1000, 1000), color_mode='grayscale')
            image = gan_preprocessing(krs_image.img_to_array(image))

            manual_features = manual_dict[image_name][20:21]

            batch['images'].append(image)
            batch['manual'].append(manual_features)
            i += 1

        batch['images'] = np.array(batch['images'])
        batch['manual'] = np.array(batch['manual'])

        yield batch['images'], batch['manual']

# ------------------------------------------------------------------ FEATURES EXTRACTION --------------------------------------------------------------------------
def get_hand_crafted(one_image):
    """ Extracts various features out of the given image
    :param array one_image: the image from which features are extracted
    :return: the features associated with this image
    :rtype: Numpy array of size (34, 1)
    """
    hist = histogram(one_image, nbins=20, normalize=True)
    features = hist[0]
    blob_lo = blob_log(one_image, max_sigma=2.5, min_sigma=1.5, num_sigma=5, threshold=0.05)
    shape_ind = shape_index(one_image)
    shape_hist = np.histogram(shape_ind, range=(-1, 1), bins=9)
    shan_ent = shannon_entropy(one_image)
    max_val = one_image.max()
    min_val = one_image.min()
    variance_val = np.var(one_image)
    features = np.concatenate([features, [blob_lo.shape[0]], shape_hist[0], [shan_ent], [max_val], [min_val], [variance_val]])
    return features

def features_summary(image_set, decode=True, return_ids=True):
    features = []
    ids = []
    for image in tqdm(image_set):
        if return_ids:
            ids.append(int(str(image).split("/")[-1].split(".")[0].split("_")[-1]))
        if decode:
            image = color.rgb2gray(io.imread(image))
            image = vanilla_preprocessing(image)
        assert np.amax(image) <= 1 and np.amin(image) >= 0 # Image must be in the same range to be compared
        features.append(get_hand_crafted(image))
    features = np.array(features)

    # Compute mean and variance of the features
    mean_features = np.mean(np.copy(features), axis=0)
    var_features = np.var(np.copy(features), axis=0)

    return features, mean_features, var_features, np.array(ids)

def extract_and_save_features(image_dir, prefix, out_dir="manual_features", max_imgs=None):
    """
    Extract manual features from images contained in dir and saves them in the out_directory
    """
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    if not os.path.isdir(os.path.join(out_dir, prefix)):
        os.mkdir(os.path.join(out_dir, prefix))

    all_images = [str(item) for item in pathlib.Path(image_dir).glob('*')]
    if max_imgs and len(all_images) > max_imgs:
        all_images = all_images[:max_imgs]
    features, means, vars, ids = features_summary(all_images, True)
    np.savetxt(os.path.join(out_dir, prefix, "features_{}.gz".format(prefix)), features)
    np.savetxt(os.path.join(out_dir, prefix,  "means_{}.gz".format(prefix)), means)
    np.savetxt(os.path.join(out_dir, prefix,  "vars_{}.gz".format(prefix)), vars)
    np.savetxt(os.path.join(out_dir, prefix,  "ids_{}.gz".format(prefix)), ids)


# ------------------------------------------------------------------ EXPERIMENTS --------------------------------------------------------------------------

def heatmap(images_set, decode=False, shape=(1000, 1000)):
    """
    Given an image set, summarized it into a mean image
    :param decode: weather the image needs to be decoded and pre-processed
    :param shape: input shape
    :return: the mean image
    """
    sum = np.zeros(shape)
    for image in images_set:
        if decode:
            image = color.rgb2gray(io.imread(image))
            image = vanilla_preprocessing(image)

        assert np.amax(image) <= 1 and np.amin(image) >= 0 # Image must be in the same range to be compared
        sum += image
    sum /= len(images_set)
    return sum

def knn_diversity_stats(training_set, generated_imgs):
    """
    Find the k=3 nearest neighnors of an image in the training set and
    returns the average distance
    """
    knn = sk.neighbors.NearestNeighbors(n_neighbors=3)
    knn.fit(training_set)

    dists, idxs = knn.neighbors(generated_imgs)
    return np.average(dists)

def make_max_pooling_resizer():
    """
    Keras resizer for resizing 1000x1000 images into 64x64 max_pooled images
    """
    resizer = tf.keras.Sequential(
        [tf.keras.layers.Lambda(lambda x: x + 1),
         tf.keras.layers.ZeroPadding2D(padding=(12, 12)),
         tf.keras.layers.Lambda(lambda x: x - 1),
         tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
         tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
         tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
         tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
         tf.keras.layers.Reshape(target_shape=(64, 64, 1))]
    )
    return resizer

# ------------------------------------------------------------------ IMAGE PREPROCESSING --------------------------------------------------------------------------

def vanilla_preprocessing(image):
    """ Normalizes the image by cliping pixels to value [0,1]
    :param array image: image to be processed
    :return : processed image
    :rtype: numpy array
    """
    image = image / 255.0
    return image


def vae_preprocessing(image):
    """Returns a random permutation of the argument image
    :param array image: subject image
    :return: rotated image
    :rtype: numpy array
    """
    rint = random.randint(1, 4)
    image = np.rot90(image, rint)
    #image = image / 255.0
    return image
    
def gan_preprocessing(image):
    """Processes the image by cliiping each pixel to value in range [-1, 1] and
    taking a random rotation
    :param array image: the image to be processed
    :return: the processed image
    :rtype : a numpy array
    """
    rint = random.randint(1, 4)
    image = np.rot90(image, rint)
    image = image / 255.0
    image = (image - 0.5) / 0.5
    return image

# ----------------------------------------------- GET TRAIN/PREDICTION DATA (VAE ENCODED IMAGES AND/OR FEATURES) ---------------------------------------------------------------

def encode_scored_images(vae, generator, save_dir, only_features=False, feature_dim=34, latent_dim=100, batch_size=16):
    """ Encodes every image in the generator in the latent space of the vae and saves each
    the encoded images concatenated with the features and scores to a .npy file
    :param model vae: the vae used to encode the images
    :param generator generator: the container yielding elements of size batch_size
    and containing the images to be encoded, the associated features and scores
    :param only_features: true if the regressor must trained only on features
    :param int feature_dim: the number of features
    :param int latent_dim: the dimension of the vae latent space
    :param int batch_size: the batch size to be used
    """
    start_time = t.time()

    print("Creating dataset for training...")
 
    i = 0
    train_data, score_data = [], []
    
    for (X_batch, features), score_batch in generator:
        if not only_features:
            mean, logvar = vae.encode(X_batch)
            z = vae.reparameterize(mean, logvar)      
            z = np.reshape(np.clip(np.reshape(z, (-1)),-1e15, 1e15), (batch_size, latent_dim))
            z = np.concatenate([z, features], axis=1)
            train_data.append(z)
        else: 
            train_data.append(features)
        
        score_data.append(score_batch)
      
        if i % 50 == 0:
            print("progression: Iteration: ", i, "time since start: ", \
              t.time() - start_time ," s ")
        i = i + 1
    
    print("Saving computations..")
    if not only_features:
        train_data = np.array(train_data).reshape((-1, latent_dim + feature_dim))
        file_name = os.path.join(save_dir, "scored_image_" + str(latent_dim))
    else:
        train_data = np.array(train_data).reshape((-1, feature_dim))
        file_name = os.path.join(save_dir, "scored_image_only_features")
      
    np.save(file_name, train_data)
    score_data = np.array(score_data).reshape((-1))
    np.save(os.path.join(save_dir, "scores"), score_data)
  
def encode_query_images(vae, generator, save_dir, only_features=False, feature_dim=34, latent_dim=100):
    """ Encodes every image in the generator in the latent space of the vae and saves
    the encoded images concatenated with the features to a .npy file
    :param model vae: the vae used to encode the images
    :param generator generator: the container yielding elements of size batch_size
    and containing the images to be encoded, the associated features
    :param only_features: true if the regressor will predict only based on features    
    :param int feature_dim: the number of features
    :param int latent_dim: the dimension of the vae latent space
    :param int batch_size: the batch size to be used
    """
    start_time = t.time()
    
    print("Creating dataset for predictions...")

    i = 0
    queries = []
    for (query, features), dummy in generator:
        if not only_features:
            mean, logvar = vae.encode(query)
            z = vae.reparameterize(mean, logvar)  
            z = np.concatenate([z, features], axis = 1)
            queries.append(z)
        else:
            queries.append(features)
    
        if i %50 == 0:
            print("iteration :", i, ", time passed", t.time() - start_time ," s")
        i+=1
    
    print("Saving computations ...")
    if not only_features:
        queries = np.array(queries).reshape((-1, latent_dim + feature_dim))
        file_name = os.path.join(save_dir, "query_image_" + str(latent_dim))
    else:
        queries = np.array(queries).reshape((-1, feature_dim))
        file_name = os.path.join(save_dir, "query_image_only_features")

    np.save(file_name , queries)

# ----------------------------------------------- FUNCTIONS FOR TRAINING REGRESSOR AND PREDICTING SCORES ON QUERY ---------------------------------------------------------------

def train_regressor(regr, vae, generator, save_root, only_features=False,
                   feature_dim=34, latent_dim=100, batch_size=16):
    """ Trains a regressor on the images, features and scores contained in the generator
    :param sklearn regressor regr: the regressor to be trained
    :param model vae: the vae used to encode the images in its latent space
    :param generator generator: the container yielding elements of size batch_size
    and containing the images to be encoded, the associated features and scores
    :param only_features: true if the regressor must trained only on features
    :param int feature_dim: the number of features
    :param int latent_dim: the dimension of the vae latent space
    :param int batch_size: the batch size to be used
    :returns: the trained regressor
    :rtype: sklearn regressor
    """
  
    train_data_dir = os.path.join(save_root, "train_data")
    
    if not os.path.exists(train_data_dir): # if directory exists, then skip data encoding
        os.makedirs(train_data_dir)
        print("\nCreating training data and saving at {}".format(train_data_dir))
        encode_scored_images(vae, generator, train_data_dir, only_features=only_features, 
                             feature_dim=feature_dim, latent_dim=latent_dim, batch_size=batch_size)
        print("Done.")
    else:
        print("Training data dir exists, skipping data encoding.")
        
    if only_features:
        filename = os.path.join(train_data_dir, "scored_image_only_features.npy")
    else:
        filename = os.path.join(train_data_dir, "scored_image_{}.npy".format(latent_dim))
    
    # load training data
    train_data = np.load(filename)
    score_data = np.load(os.path.join(train_data_dir, "scores.npy"))
  
    print("Finished encoding/loading: training data", np.shape(train_data), "score_data:", np.shape(score_data))
    print("Fitting data..")
    regr.fit(train_data, score_data)
        
    return regr
              
def predict(vae, regr, generator, save_root, query_numbers, only_features=False, feature_dim=34, latent_dim=100):
    """Produces a .csv file containing the predictions of the regressor of the scores associated 
    with the images and features in the generator
    :param model vae: the vae used to encode the images in its latent space
    :param sklearn regressor regr: the regressor used to make predictions
    :param generator generator: the container yielding elements of size batch_size
    and containing the images to be encoded and the associated features
    :param query_numbers: contains the ids of the images yielded by the generator 
    in the same order
    :param only_features: true if the regressor must predict only based on features
    :param int feature_dim: the number of features
    :param int latent_dim: the dimension of the vae latent space
    """
    
    predictions = make_predictions(vae, regr, generator, save_root, only_features=only_features, 
                                   feature_dim=feature_dim, latent_dim=latent_dim)
    predictions = np.clip(predictions, a_min=0, a_max=8)
    
    predictions = np.array(predictions).reshape((-1, 1))
    print("Predictions size", np.shape(predictions))
    query_numbers = np.array(query_numbers).reshape((-1, 1))
    print("Query_numbers size", np.shape(query_numbers))
    
    indexed_predictions = np.concatenate([query_numbers, predictions], axis=1)
    print("Saving to .csv...")
    
    pred_save_dir = os.path.join(save_root, "predictions")
    
    if not os.path.exists(pred_save_dir):
        os.makedirs(pred_save_dir)
    
    np.savetxt(os.path.join(pred_save_dir, "predictions.csv"), indexed_predictions, header='Id,Predicted', delimiter=",", fmt='%d, %f', comments="")
        
        
def make_predictions(regr, vae, generator, save_root, only_features=False, feature_dim=34, latent_dim=100):
    """ Returns an array containing the predicted scores of the images yielded 
    by the generator
    :param sklearn regressor regr: the regressor used to make predictions
    :param model vae: the vae used to encode the images in its latent space
    :param generator generator: the container yielding elements of size batch_size
    and containing the images to be encoded and the associated features
    :param only_features: true if the regressor must predict only based on features    
    :param int feature_dim: the number of features
    :param int latent_dim: the dimension of the vae latent space
    """
    pred_data_dir = os.path.join(save_root, "pred_data")
    
    if not os.path.exists(pred_data_dir): # if directory exists, then skip data encoding
        os.makedirs(pred_data_dir)
        encode_query_images(vae, generator, pred_data_dir, only_features=only_features, 
                            feature_dim=feature_dim, latent_dim=latent_dim)
    else:
        print("Predictions data dir exists, skipping data encoding.")
    
    if only_features:
        filename = os.path.join(pred_data_dir, "query_image_only_features.npy")
    else:
        filename = os.path.join(pred_data_dir, "query_image_{}.npy".format(latent_dim))

    # load features for prediction
    queries = np.load(filename)
    predictions = np.array(regr.predict(queries))
    print("Finished predicting: predictions", np.shape(predictions))
        
    return predictions
    
# ----------------------------------------------- MAIN FUNCTION CALLED FOR TRAINING AND PREDICTING ---------------------------------------------------------------
  
def predict_with_regressor(vae, regr_type, scored_feature_generator, query_feature_generator,
                           sorted_queries, only_features=False, feature_dim=34, 
                           latent_dim=100, batch_size=16, random_state=10, save_root="./Regressor"):
    
    """ creates a .csv file containing the predictions
    of the regressor on the query images
    :param model vae: the vae used to encode the images in its latent space
    :param str regr_type: the type of sklearn regressor to used (Options: Random_Forest, Ridge, MLP, Boost)
    :param generator scored_feature_generator: the container yielding elements of size batch_size
    and containing the images to be encoded, the associated features and the scores
    :param generator query_feature_generator: the container yielding elements of size batch_size
    and containing the images to be encoded and the associated features
    :param sorted_queries: contains the ids of the images yielded by the query generator 
    in the right order
    :param only_features: true if the regressor must predict only based on features    
    :param int feature_dim: the number of features
    :param int latent_dim: the dimension of the vae latent space
    :param int batch_size: the batch size to be used
    :param int random_state: the number of random states to used in the case of 
    a random forest regressor
    :param str save_path: the path to which the model of the regressor will be saved  
    """
    if regr_type == "Random_Forest": 
      
      base_model = RandomForestRegressor(criterion="mae", max_features=None, oob_score=True,
    												random_state = random_state) 
      regr = model_selection.GridSearchCV(base_model, {"n_estimators": [50, 100], "max_depth": [16, 32]},
                                          verbose=5, scoring='neg_mean_absolute_error') 
    
    elif regr_type == "Ridge":
      
      base_model = linear_model.Ridge()
      regr = model_selection.GridSearchCV(base_model, {"alpha": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100, 1000]}, verbose=5, scoring= 'neg_mean_absolute_error' )      
    
    elif regr_type == "MLP" :
            
      base_model = neural_network.MLPRegressor()
      regr = model_selection.GridSearchCV(base_model, param_grid={
      'learning_rate_init': [0.05, 0.01, 0.005, 0.001],
      "alpha": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100, 1000], 
      "hidden_layer_sizes": [100, 500, 1000],
      'power_t': [0.5, 0.2, 0.8],
      "activation" : ["identity", "logistic", "tanh", "relu"]})
    
    elif regr_type == "Boost":
      base_model = xgb.XGBRegressor()
      regr = model_selection.GridSearchCV(base_model, {"colsample_bytree":[1.0],"min_child_weight":[1.0,1.2],
                            'max_depth': [3,8, 16], 'n_estimators': [500,1000]}, verbose=1)
    else:
      raise NameError("Unknown regressor type !")
    
    print("\n--Training regressor {} --".format(regr_type))
    regr = train_regressor(regr, vae, scored_feature_generator, save_root, 
                           only_features=only_features, feature_dim=feature_dim,
                           batch_size=batch_size, latent_dim=latent_dim)
    
    print("\n--Creating prediction.csv file--")
    predict(regr, vae, query_feature_generator, save_root, sorted_queries, 
            only_features=only_features, feature_dim=feature_dim, latent_dim=latent_dim)
    
    checkpoints_dir = os.path.join(save_root, "checkpoints")
    if(not os.path.isdir(checkpoints_dir)):
      os.mkdir(checkpoints_dir)


    joblib.dump(regr, os.path.join(checkpoints_dir, regr_type))

# ---------------------------------------------------------- SPLIT LABELED IMAGES TO 2 FOLDERS BY LABEL --------------------------------------------------------------------------

def create_labeled_folders(data_root):
    """Creates two folders within the labeled folder of the data_root folder,
    separating images associated to a label of 0 and of 1
    
    param str data_root: Path of the folder containing labeled and scored folders and csv files
    """
    labeled_images_new_path = os.path.join(data_root, "labeled_split")
    if not os.path.exists(labeled_images_new_path):
        os.makedirs(labeled_images_new_path)
    else:
        print("\nDirectory labeled_split exists, nothing to do ...")
        return
    
    print("\nSplitting data labeled 0/1 to 2 folders labeled_split/0 and labeled_split/1")
    
    labels_path = os.path.join(data_root, "labeled.csv")
    labels = pd.read_csv(labels_path, index_col=0, skiprows=1, header=None)
    id_to_label = labels.to_dict(orient="index")
    id_to_label = {k: v[1] for k, v in id_to_label.items()}

    labeled_images_path = os.path.join(data_root, "labeled")
    labeled_images_path = pathlib.Path(labeled_images_path)
    onlyFiles = [f for f in os.listdir(labeled_images_path) if (os.path.isfile(os.path.join(labeled_images_path, f)) and (f != None))]
    all_indexes = [item.split('.')[0] for item in onlyFiles]
    all_indexes = filter(None, all_indexes)
    all_pairs = [[item, id_to_label[int(item)]] for item in all_indexes]

    
    # Add if does not exist
    if(~os.path.isdir(os.path.join(labeled_images_new_path, '0'))):
        os.mkdir(os.path.join(labeled_images_new_path, '0'))
    if(~os.path.isdir(os.path.join(labeled_images_new_path, '1'))):
        os.mkdir(os.path.join(labeled_images_new_path, '1'))

    print("Copying files to folders labeled_split/0 and labeled_split/1")
    for file, label in all_pairs:
        copyfile(os.path.join(labeled_images_path, file + '.png'), os.path.join(labeled_images_new_path, "{}".format(int(label)), file + '.png'))
    print("Done.")

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#def flow_from_dataframe(img_data_gen, in_df, path_col, y_col, batch_size=16, subset='training', **dflow_args):
#    base_dir = os.path.dirname(in_df[path_col].values[0])
#    print('## Ignore next message from keras, values are replaced anyways')
#    df_gen = img_data_gen.flow_from_directory(base_dir, 
#                                     class_mode = 'sparse',
#                                     batch_size=batch_size,
#                                     target_size=(1000, 1000),
#                                     subset=subset,
#                                     color_mode='grayscale',
#                                    **dflow_args)
#    df_gen.filenames = in_df[path_col].values
#    df_gen._filepaths = df_gen.filenames
#    df_gen.classes = np.stack(in_df[y_col].values)
#    df_gen.samples = in_df.shape[0]
#    df_gen.n = in_df.shape[0]
#    df_gen.directory = '' # since we have the full path
#    df_gen._set_index_array()        
#    print('Reinserting dataframe: {} images'.format(in_df.shape[0]))
#    return df_gen
