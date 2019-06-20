import random
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
import time as t


def gan_preprocessing(image):
    image = image / 255.0
    image = (image - 0.5) / 0.5
    return image


def vae_preprocessing(image):
    rint = random.randint(1, 4)
    image = np.rot90(image, rint)
    image = image / 255.0
    return image

def train_random_forest_regressor(vae , epochs, scored_generator_train,\
                                  latent_dim=100, batch_size=16, max_depth=2, \
                                  random_state=0, nb_trees=100, iteration_limit=None): 
  
  
  regr = RandomForestRegressor(max_depth=max_depth, random_state=random_state)
  start_time = t.time()
  for epoch in range(epochs):
    for i, scored_image in enumerate(scored_generator_train):
      X_batch = scored_image[0]
      score_batch = scored_image[1]
    
      mean, logvar = vae.encode(X_batch)
      z = vae.reparameterize(mean, logvar)
      ##Clipping the values..
      z = np.reshape(np.clip(np.reshape(z, (-1)),tf.float32.min, tf.float32.max), (batch_size, latent_dim))
        
      regr.fit(z, score_batch)  
        
      if iteration_limit != None and i >= iteration_limit:
        return regr
        
      if i % 50 == 0:
        print("progression: epoch, ", epoch, ", iteration: ", i, "time since start: ", \
              t.time() - start_time ," s, " , i/min(iteration_limit, len(scored_generator_train))*10, "%")
        
  return regr
              
def predict_with_random_forest_regressor(vae, regr, query_generator, query_numbers):
    
        predictions = make_predictions(vae, regr, query_generator)
        predictions = np.clip(predictions, a_min=0, a_max=8)
        
        predictions = np.array(predictions).reshape((-1, 1))
        print("Predictions size", np.shape(predictions))
        query_numbers = np.array(query_numbers).reshape((-1, 1))
        print("Query_numbers size", np.shape(query_numbers))
        
        indexed_predictions = np.concatenate([query_numbers, predictions], axis=1)
        print("Saving to .csv...")
        np.savetxt("predictions.csv", indexed_predictions, header='Id,Predicted', delimiter=",", fmt='%d, %f', comments="")
        
        
def make_predictions(vae, regr, query_generator):
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