import glob
import numpy as np 
import os, sys
import time, datetime
import xgboost as xgb
from sklearn import model_selection
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-rt', '--regressor_type', type = str, default = "Boost", choices=["Random_Forest", "Ridge", "Boost"], 
                        help = 'type of regressor trained on the manually extracted features and used to make predictions on the query image dataset')

args = parser.parse_args()

REGRESSOR_TYPE=args.regressor_type

DATA_ROOT = "./data"
FEATURES_DIR = os.path.join(DATA_ROOT, "features")

print("\n")
print("Run infos:")
print("    REGRESSOR_TYPE: {}".format(REGRESSOR_TYPE))
print("\n")
sys.stdout.flush()

#sys.exit(0)

SCORED_FEATS_PATH = os.path.join(FEATURES_DIR, 'scored_feats.gz')
SCORED_FEATS_IDS_PATH = os.path.join(FEATURES_DIR, 'scored_feats_ids.gz')

QUERY_FEATS_PATH = os.path.join(FEATURES_DIR, 'query_feats.gz')
QUERY_FEATS_IDS_PATH = os.path.join(FEATURES_DIR, 'query_feats_ids.gz')

def timestamp():
    return datetime.datetime.fromtimestamp(time.time()).strftime("%Y.%m.%d-%H:%M:%S")
    
def get_regressor(regr_type="Boost", random_state=10):
    if regr_type == "Random_Forest": 
      base_model = RandomForestRegressor(criterion="mae", max_features=None, oob_score=True,
												    random_state = random_state) 
      regr = model_selection.GridSearchCV(base_model, {"n_estimators": [50, 100], "max_depth": [16, 32]},
                                          verbose=5, scoring='neg_mean_absolute_error', n_jobs=-1) 

    elif regr_type == "Ridge":
      base_model = linear_model.Ridge()
      regr = model_selection.GridSearchCV(base_model, {"alpha": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100, 1000]}, verbose=5, scoring= 'neg_mean_absolute_error' )      

    elif regr_type == "Boost":
      base_model = xgb.XGBRegressor()
      regr = model_selection.GridSearchCV(base_model, {"colsample_bytree":[1.0],"min_child_weight":[1.0,1.2],
                            'max_depth': [3,8, 16], 'n_estimators': [500,1000]}, verbose=1)
    return regr

def main():
    """ 
    The method that takes care of running the model and predicting scores of the query images
    """

    print("Start at {} ...".format(timestamp()))


#    query_dir_list = glob.glob(os.path.join(DATA_ROOT, "query", "*"))
#    only_files = [file_path for file_path in query_dir_list if (os.path.isfile(file_path)]
#    all_indexes = [int(item.split('/')[-1].split('.')[0]) for item in only_files]
#    sorted_queries = np.sort(all_indexes)

    if not os.path.exists(FEATURES_DIR):
        print("FEATURES_DIR: {} not found".format(FEATURES_DIR))
        sys.exit(-1)
    
    if not os.path.exists(SCORED_FEATS_PATH) or not os.path.exists(SCORED_FEATS_IDS_PATH):
        print("Scored dataset features {} or ids {} not found, please run generate_feats.py ...".format(SCORED_FEATS_PATH, SCORED_FEATS_IDS_PATH))
        sys.exit(-1)
        
    if not os.path.exists(QUERY_FEATS_PATH) or not os.path.exists(QUERY_FEATS_IDS_PATH):
        print("Scored dataset features {} or ids {} not found, please run generate_feats.py ...".format(QUERY_FEATS_PATH, QUERY_FEATS_IDS_PATH))
        sys.exit(-1)

    scored_feats_with_scores = np.loadtxt(SCORED_FEATS_PATH)
    scored_feats = scored_feats_with_scores[:, :-1] # take all except the last column
    scored_scores = scored_feats_with_scores[:, -1] # extract the last column
    scored_ids = np.loadtxt(SCORED_FEATS_IDS_PATH).astype(int)
    
    query_feats = np.loadtxt(QUERY_FEATS_PATH)
    query_ids = np.loadtxt(QUERY_FEATS_IDS_PATH).astype(int)

    nb_feats_scored = scored_feats.shape[1]
    nb_feats_query = query_feats.shape[1]
    assert nb_feats_query == nb_feats_scored
    
#    FEATURES_DIM=nb_feats_query
    
    print("\nShape manual scored features", scored_feats.shape)
    print("Shape scores", scored_scores.shape)
    print("Shape manual scored ids", scored_ids.shape)
    
    print("Shape manual query features", query_feats.shape)
    print("Shape manual query ids", query_ids.shape)


#    scored_dict = dict(zip(scored_ids, scored_feats))
#    query_dict = dict(zip(query_ids, query_feats))

#    image_score_list = [os.path.join(DATA_ROOT, "scored", str(i) + ".png") for i in scored_ids]
#    scored_feature_generator = utils.custom_generator(image_score_list, manual_score_dict, FEATURES_DIM)

#    image_query_list = [os.path.join(DATA_ROOT, "query", str(i) + ".png") for i in manual_query_ids]
#    query_feature_generator = utils.custom_generator(image_query_list, manual_query_dict, FEATURES_DIM, do_score= False, batch_size=1)

#    print("Data generators have been created")

    #    sys.exit(0)

    if REGRESSOR_TYPE is not None:
        regr = get_regressor(REGRESSOR_TYPE)
        
        REGRESSOR_DIR = os.path.join("./Regressor", REGRESSOR_TYPE)
        if not os.path.exists(REGRESSOR_DIR):
            os.makedirs(REGRESSOR_DIR)
        
        print("\n--Training regressor {} --".format(REGRESSOR_TYPE))
        regr.fit(scored_feats, scored_scores)
        
        print("\n--Creating prediction.csv file--")
        query_dict = dict(zip(query_ids, query_feats))
        query_ids_sorted = np.sort(query_ids)
        query_feats_sorted = np.array([query_dict[id] for id in query_ids_sorted])
        
        predictions = np.array(regr.predict(query_feats_sorted))
        predictions = np.clip(predictions, a_min=0, a_max=8).reshape((-1, 1))
        print("Predictions shape:", predictions.shape)
        query_ids = query_ids_sorted.reshape([-1, 1])
        print("Query ids shape:", query_ids.shape)
        indexed_predictions = np.concatenate([query_ids, predictions], axis=1)
        
        pred_save_dir = os.path.join(REGRESSOR_DIR, "predictions")
    
        if not os.path.exists(pred_save_dir):
            os.makedirs(pred_save_dir)
        
        np.savetxt(os.path.join(pred_save_dir, "predictions.csv"), indexed_predictions, header='Id,Predicted', delimiter=",", fmt='%d, %f', comments="")
        
        checkpoints_dir = os.path.join(REGRESSOR_DIR, "checkpoints")
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)

        print("\n--Saving model checkpoint--")
        joblib.dump(regr, os.path.join(checkpoints_dir, REGRESSOR_TYPE+".ckpt"))
        

    print("End at {} ...".format(timestamp()))
    
if __name__ == '__main__':
    main()


