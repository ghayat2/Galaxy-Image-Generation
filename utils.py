import random
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model, neural_network
import tensorflow as tf
import time as t
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense

from sklearn.datasets import load_sample_image
from sklearn.feature_extraction import image
from skimage.feature import blob_doh, blob_log
from skimage.exposure import histogram
from skimage.feature import shape_index
from skimage.measure import shannon_entropy


def get_hand_crafted(one_image):
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

def gan_preprocessing(image):
    rint = random.randint(1, 4)
    image = np.rot90(image, rint)
    image = image / 255.0
    image = (image - 0.5)/0.5
    return image

def vanilla_preprocessing(image):
    image = image / 255.0
    return image


def vae_preprocessing(image):
    rint = random.randint(1, 4)
    image = np.rot90(image, rint)
    #image = image / 255.0
    return image

def train_regressor(regr, vae , epochs, scored_generator_train,\
                                  latent_dim=100, batch_size=16, max_depth=2, \
                                  random_state=0, nb_trees=100, iteration_limit=None): 
  
  start_time = t.time()
  for epoch in range(epochs):
    i = 0
    
    for X_batch, score_batch in scored_generator_train:
    
      mean, logvar = vae.encode(X_batch)
      z = vae.reparameterize(mean, logvar)  
      z = np.reshape(np.clip(np.reshape(z, (-1)),-1e15, 1e15), (batch_size, latent_dim))
      regr.partial_fit(z, score_batch) if regr == "MLP" else regr.fit(z, score_batch)
        
      if iteration_limit and i >= iteration_limit:
        return regr
        
      if i % 50 == 0:
        if not iteration_limit:
          iteration_limit = len(scored_generator_train)
        print("progression: epoch, ", epoch, ", iteration: ", i, "time since start: ", \
              t.time() - start_time ," s, " , i/min(iteration_limit, len(scored_generator_train))*100, "%")
        
      i = i + 1
      
      if i >= len(scored_generator_train):
        break
        
  return regr
              
def predict(vae, regr, query_generator, query_numbers):
    
        predictions = make_predictions(vae, regr, query_generator)
        predictions = np.clip(predictions, a_min=0, a_max=8)
        
        predictions = np.array(predictions).reshape((-1, 1))
        print("Predictions size", np.shape(predictions))
        query_numbers = np.array(query_numbers).reshape((-1, 1))
        print("Query_numbers size", np.shape(query_numbers))
        
        indexed_predictions = np.concatenate([query_numbers, predictions], axis=1)
        print("Saving to .csv...")
        np.savetxt("predictions.csv", indexed_predictions, header='Id,Predicted', delimiter=",", fmt='%d, %f', comments="")
        
        
def make_predictions(regr, vae, query_generator):
    predictions = []
    start_time = t.time()
    i = 0
    for query in query_generator:
        mean, logvar = vae.encode(query)
        z = vae.reparameterize(mean, logvar)  
        predictions.append(regr.predict(z))
        if i %50 == 0:
          print("iteration :", i, ", time passed", t.time() - start_time ," s,", "progression: ", i/len(query_generator)*100, "%")
        if i >= len(query_generator)-1:
          return predictions
        i = i + 1
        
    return predictions
  
def predict_with_regressor(vae, regr_type, scored_generator_train, query_generator,
                           sorted_queries, alpha=0.5, epochs=10, 
                           batch_size=16, max_depth=2, random_state=0):
    
    if regr_type == "Random Forest":      
        regr = RandomForestRegressor(max_depth=max_depth, random_state=random_state)
    elif regr_type == "Ridge":
        regr = linear_model.Ridge(alpha=alpha)
    elif regr_type == "MLP":
        regr = neural_network.MLPRegressor(batch_size=batch_size)
    else :
        raise NameError("Unknown regressor type !")

    print("Training regressor...")
    regr = train_regressor(regr, vae, epochs, scored_generator_train,
                                         batch_size=batch_size)
    
    print("Creating prediction.csv file...")
    predict(regr, vae, query_generator, sorted_queries)
    