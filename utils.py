import random
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model, neural_network
import tensorflow as tf
import time as t
import random

from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense
from keras.preprocessing import image as krs_image

from sklearn import model_selection
from sklearn.externals import joblib
from sklearn.datasets import load_sample_image
from sklearn.feature_extraction import image
from skimage.feature import blob_doh, blob_log
from skimage.exposure import histogram
from skimage.feature import shape_index
from skimage.measure import shannon_entropy


def custom_generator(images_list, manual_dict, score=True, batch_size=16):
    """ Returns an numpy array of size batch_size containing the decoded images,
    the features associated with these images and their scores.
    
    :param array image_list: An array containing the full path of the images
    :param dict manual_dict: A dictionary associating the id of an image to 
    its features ([:34]) and its score ([35])
    :param bool score: Wether the images are scored or not
    :param int batch_size: The batch_size to use
    :rtype: Numpy array
    """
    i = 0
    images_list = np.sort(images_list)
    
    while i<len(images_list):
        batch = {'images': [], 'manual': [], 'scores': []}
        for b in range(batch_size):
            # Read image from list and convert to array
            image_path = images_list[i]
            image_name = int(os.path.basename(image_path).replace('.png', ''))
            image = krs_image.load_img(image_path, target_size=(1000, 1000), color_mode='grayscale')
            image = gan_preprocessing(krs_image.img_to_array(image))

            manual_features = manual_dict[image_name][:34]

            score = manual_dict[image_name][34] if score else 0

            batch['images'].append(image)
            batch['manual'].append(manual_features)
            batch['scores'].append(score)

            i += 1

        batch['images'] = np.array(batch['images'])
        batch['manual'] = np.array(batch['manual'])
        batch['scores'] = np.array(batch['scores'])

        yield [batch['images'], batch['manual']], batch['scores']
        
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


def get_hand_crafted(one_image):
    """ Extracts various features out of the given image
    :param array one_image: the image from which features are extracted
    :return: the features associated with this image
    :rtype: Numpy array of size (34, 1)
    """
    hist = histogram(one_image, nbins=20, normalize=True)
    features = hist[0]
    blob_lo = blob_log(one_image, max_sigma=2.5, min_sigma=1.5, num_sigma=30, threshold=0.05)
    shape_ind = shape_index(one_image)
    shape_hist = np.histogram(shape_ind, range=(-1, 1), bins=9)
    shan_ent = shannon_entropy(one_image)
    max_val = one_image.max()
    min_val = one_image.min()
    variance_val = np.var(one_image)
    features = np.concatenate([features, [blob_lo.shape[0]], shape_hist[0], [shan_ent], [max_val], [min_val], [variance_val]])
    return features

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


def encode_scored_images(vae, generator, feature_dim=34, latent_dim=100, batch_size=16):
    """ Encodes every image in the generator in the latent space of the vae and saves each
    the encoded images concatenated with the features and scores to a .npy file
    :param model vae: the vae used to encode the images
    :param generator generator: the container yielding elements of size batch_size
    and containing the images to be encoded, the associated features and scores
    :param int feature_dim: the number of features
    :param int latent_dim: the dimension of the vae latent space
    :param int batch_size: the batch size to be used
    """
    start_time = t.time()
      
    print("Encoding training data in latent space..")
 
    i = 0
    train_data, score_data = [], []
    
    for (X_batch, features), score_batch in generator:
      mean, logvar = vae.encode(X_batch)
      z = vae.reparameterize(mean, logvar)      
      z = np.reshape(np.clip(np.reshape(z, (-1)),-1e15, 1e15), (batch_size, latent_dim))
      z = np.concatenate([z, features], axis=1)
      train_data.append(z)
      score_data.append(score_batch)
      
      if i % 50 == 0:
        print("progression: Iteration: ", i, "time since start: ", \
              t.time() - start_time ," s ")
      i = i + 1
   
    print("Saving latent space image representation..")
    train_data = np.array(train_data).reshape((-1, latent_dim + feature_dim))
    np.save("scored_images_" + str(latent_dim), train_data)
    score_data = np.array(score_data).reshape((-1))
    np.save("scores", score_data)
  
def encode_query_images(vae, generator, feature_dim=34, latent_dim=100):
    """ Encodes every image in the generator in the latent space of the vae and saves
    the encoded images concatenated with the features to a .npy file
    :param model vae: the vae used to encode the images
    :param generator generator: the container yielding elements of size batch_size
    and containing the images to be encoded, the associated features
    :param int feature_dim: the number of features
    :param int latent_dim: the dimension of the vae latent space
    :param int batch_size: the batch size to be used
    """
    start_time = t.time()
    
    print("Encoding query data in latent space..")

    i = 0
    queries = []
    for (query, features), dummy in generator:
        mean, logvar = vae.encode(query)
        z = vae.reparameterize(mean, logvar)  
        z = np.concatenate([z, features], axis = 1)
        queries.append(z)
    
    if i %50 == 0:
      print("iteration :", i, ", time passed", t.time() - start_time ," s,", "progression: ")
    i+=1
    
    print("Saving latent space image representation..")
    queries = np.array(queries).reshape((-1, latent_dim + feature_dim))
    np.save("query_images_" + str(latent_dim), queries)

def train_regressor(regr, vae , generator, vae_encoded_images=False, 
                   feature_dim=34, latent_dim=100, batch_size=16):
    """ Trains a regressor on the images, features and scores contained in the generator
    :param sklearn regressor regr: the regressor to be trained
    :param model vae: the vae used to encode the images in its latent space
    :param generator generator: the container yielding elements of size batch_size
    and containing the images to be encoded, the associated features and scores
    :param bool vae_encoded_images: true if .npy files for the encoded images and scores 
    already exists
    :param int feature_dim: the number of features
    :param int latent_dim: the dimension of the vae latent space
    :param int batch_size: the batch size to be used
    :returns: the trained regressor
    :rtype: sklearn regressor
    """
  
    if not vae_encoded_images:
        encode_scored_images(vae, generator, feature_dim=feature_dim, latent_dim=latent_dim, batch_size=batch_size)
    
    train_data = np.load("scored_images_" + str(latent_dim) + ".npy")
    score_data = np.load("scores.npy")
  
    print("Finished encoding: training data", np.shape(train_data), "score_data:", np.shape(score_data))
    print("Fitting data..")
    regr.fit(train_data, score_data)
        
    return regr
              
def predict(vae, regr, generator, query_numbers, vae_encoded_images=False, feature_dim=34, latent_dim=100):
    """Produces a .csv file containing the predictions of the regressor of the scores associated 
    with the images and features in the generator
    :param model vae: the vae used to encode the images in its latent space
    :param sklearn regressor regr: the regressor used to make predictions
    :param generator generator: the container yielding elements of size batch_size
    and containing the images to be encoded and the associated features
    :param query_numbers: contains the ids of the images yielded by the generator 
    in the same order
    :param bool vae_encoded_images: true if .npy files for the encoded images
    already exists
    :param int feature_dim: the number of features
    :param int latent_dim: the dimension of the vae latent space
    """
    
    predictions = make_predictions(vae, regr, generator, 
                                   vae_encoded_images=vae_encoded_images, feature_dim=feature_dim, latent_dim=latent_dim)
    predictions = np.clip(predictions, a_min=0, a_max=8)
    
    predictions = np.array(predictions).reshape((-1, 1))
    print("Predictions size", np.shape(predictions))
    query_numbers = np.array(query_numbers).reshape((-1, 1))
    print("Query_numbers size", np.shape(query_numbers))
    
    indexed_predictions = np.concatenate([query_numbers, predictions], axis=1)
    print("Saving to .csv...")
    np.savetxt("predictions.csv", indexed_predictions, header='Id,Predicted', delimiter=",", fmt='%d, %f', comments="")
        
        
def make_predictions(regr, vae, generator, vae_encoded_images=False, feature_dim=34, latent_dim=100):
    """ Returns an array containing the predicted scores of the images yielded 
    by the generator
    :param sklearn regressor regr: the regressor used to make predictions
    :param model vae: the vae used to encode the images in its latent space
    :param generator generator: the container yielding elements of size batch_size
    and containing the images to be encoded and the associated features
    :param bool vae_encoded_images: true if .npy files for the encoded images
    already exists
    :param int feature_dim: the number of features
    :param int latent_dim: the dimension of the vae latent space
    """
     
    if not vae_encoded_images:
        encode_query_images(vae, generator, feature_dim=feature_dim, latent_dim=latent_dim)
           
    queries = np.load("query_images_" + str(latent_dim)+ ".npy")
    predictions = np.array(regr.predict(queries))
    print("Finished predicting: predictions", np.shape(predictions))
        
    return predictions
  
def predict_with_regressor(vae, regr_type, scored_feature_generator, query_feature_generator,
                           sorted_queries, vae_encoded_images=False, feature_dim=34, 
                           latent_dim=100, batch_size=16,
                           random_state=10, data_path="./Regressor/"):
    
    """ main function called from main.py to create a .csv file containing the predictions
    of the regressor on the query images
    :param model vae: the vae used to encode the images in its latent space
    :param str regr_type: the type of sklearn regressor to used (Options: Random Forest, Ridge, MLP)
    :param generator scored_feature_generator: the container yielding elements of size batch_size
    and containing the images to be encoded, the associated features and the scores
    :param generator query_feature_generator: the container yielding elements of size batch_size
    and containing the images to be encoded and the associated features
    :param sorted_queries: contains the ids of the images yielded by the query generator 
    in the right order
    :param bool vae_encoded_images: true if .npy files for the encoded images
    already exists
    :param int feature_dim: the number of features
    :param int latent_dim: the dimension of the vae latent space
    :param int batch_size: the batch size to be used
    :param int random_state: the number of random states to used in the case of 
    a random forest regressor
    :param str data_path: the path to which the model of the regressor will be saved 
    :raises: NameError exception
    """
    if regr_type == "Random Forest": 
      
      base_model = RandomForestRegressor(criterion="mae", max_features=None, oob_score=True,
    												random_state = random_state) 
      regr = model_selection.GridSearchCV(base_model, {"n_estimators": [5, 10, 50, 100]},
                                          verbose=5, scoring='neg_mean_absolute_error') 
    
    elif regr_type == "Ridge":
      
      base_model = linear_model.Ridge()
      regr = model_selection.GridSearchCV(base_model, {"alpha": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100, 1000]}, verbose=5
                                                       , scoring= 'neg_mean_absolute_error' )      
    elif regr_type == "MLP" :            
      base_model = neural_network.MLPRegressor()
      regr = model_selection.GridSearchCV(base_model, param_grid={
      'learning_rate_init': [0.05, 0.01, 0.005, 0.001],
      "alpha": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100, 1000], 
      "hidden_layer_sizes": [100, 500, 1000],
      'power_t': [0.5, 0.2, 0.8],
      "activation" : ["identity", "logistic", "tanh", "relu"]})
    else :
      raise NameError("Unknown regressor type !")
    
    print("--Training regressor--")
    regr = train_regressor(regr, vae, scored_feature_generator, vae_encoded_images=vae_encoded_images, feature_dim=feature_dim,
                                         batch_size=batch_size, latent_dim=latent_dim)
    
    print("--Creating prediction.csv file--")
    predict(regr, vae, query_feature_generator, sorted_queries, vae_encoded_images=vae_encoded_images, feature_dim=feature_dim, latent_dim=latent_dim)
          
    if(not os.path.isdir(data_path)):
      os.mkdir(data_path)
        
    joblib.dump(regr, data_path + regr_type)   