import random
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model, neural_network
import tensorflow as tf
import time as t
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense
from sklearn import model_selection
from sklearn.externals import joblib



def gan_preprocessing(image):
    rint = random.randint(1, 4)
    image = np.rot90(image, rint)
    image = image / 255.0
    return image


def vae_preprocessing(image):
    rint = random.randint(1, 4)
    image = np.rot90(image, rint)
    #image = image / 255.0
    return image



def train_regressor(regr, vae , scored_generator_train, epochs=1, \
                                  latent_dim=100, batch_size=16): 
  
  start_time = t.time()
  
  print("Encoding training data in latent space..")
  
  for epoch in range(epochs):
    i = 0
    train_data, score_data = [], []
    
    for X_batch, score_batch in scored_generator_train:
    
      mean, logvar = vae.encode(X_batch)
      z = vae.reparameterize(mean, logvar)      
      z = np.reshape(np.clip(np.reshape(z, (-1)),-1e15, 1e15), (batch_size, latent_dim))
      train_data.append(z)
      score_data.append(score_batch)
      
      if i % 50 == 0:
        print("progression: epoch, ", epoch, ", iteration: ", i, "time since start: ", \
              t.time() - start_time ," s, " , i/len(scored_generator_train)*100, "%")
        
      if i >= len(scored_generator_train):
        break
      i = i + 1
   
  train_data = np.array(train_data).reshape((-1, latent_dim))
  score_data = np.array(score_data).reshape((-1))
  print("Finished encoding: training data", np.shape(train_data), "score_data:", np.shape(score_data))
  print("Fitting data..")
  regr.fit(train_data, score_data)
        
  return regr
              
def predict(vae, regr, query_generator, query_numbers, latent_dim):
    
        predictions = make_predictions(vae, regr, query_generator, latent_dim)
        predictions = np.clip(predictions, a_min=0, a_max=8)
        
        predictions = np.array(predictions).reshape((-1, 1))
        print("Predictions size", np.shape(predictions))
        query_numbers = np.array(query_numbers).reshape((-1, 1))
        print("Query_numbers size", np.shape(query_numbers))
        
        indexed_predictions = np.concatenate([query_numbers, predictions], axis=1)
        print("Saving to .csv...")
        np.savetxt("predictions.csv", indexed_predictions, header='Id,Predicted', delimiter=",", fmt='%d, %f', comments="")
        
        
def make_predictions(regr, vae, query_generator, latent_dim):
  
    start_time = t.time()
    i = 0
    queries = []
    predictions = []
    for query in query_generator:
        mean, logvar = vae.encode(query)
        z = vae.reparameterize(mean, logvar)  
        queries.append(z)
        if i %50 == 0:
          print("iteration :", i, ", time passed", t.time() - start_time ," s,", "progression: ", i/len(query_generator)*100, "%")
        if i >= len(query_generator)-1:
          break
        i = i + 1
    
    queries = np.array(queries).reshape((-1, latent_dim))
    predictions = np.array(regr.predict(queries))
    print("Finished predicting: predictions", np.shape(predictions))

        
    return predictions
  
def predict_with_regressor(vae, regr_type, scored_generator_train, query_generator,
                           sorted_queries, latent_dim=100, epochs=1, batch_size=16,
                           random_state=10, data_path="./Regressor/"):
    
    if regr_type == "Random Forest": 
      
      base_model = RandomForestRegressor(criterion="mae", max_features=None, oob_score=True,
												random_state = random_state) 
      regr = model_selection.GridSearchCV(base_model, {"n_estimators": [5, 10, 50, 100]},
                                          verbose=5, scoring='neg_mean_absolute_error') 
    
    elif regr_type == "Ridge":
      
      base_model = linear_model.Ridge()
      regr = model_selection.GridSearchCV(base_model, {"alpha": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100, 1000]}, verbose=5, scoring= 'neg_mean_absolute_error' )      
    else :
      raise NameError("Unknown regressor type !")

    print("--Training regressor--")
    regr = train_regressor(regr, vae, scored_generator_train, epochs=epochs,
                                         batch_size=batch_size)
    
    print("--Creating prediction.csv file--")
    predict(regr, vae, query_generator, sorted_queries, latent_dim)
          
    if(not os.path.isdir(data_path)):
      os.mkdir(data_path)
        
    joblib.dump(regr, data_path + regr_type)   
