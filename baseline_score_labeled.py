import numpy as np
import sys, os, glob
import pandas as pd
import utils, data
from sklearn.externals import joblib
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-rt', '--regressor_type', type = str, default = "Boost", choices=["Random_Forest", "Ridge", "Boost"], 
                        help = 'type of regressor trained on the manually extracted features and used to make predictions on the query image dataset')
                        
args = parser.parse_args()

REGRESSOR_TYPE = args.regressor_type

REGRESSOR_DIR = os.path.join("./Regressor", REGRESSOR_TYPE)
CHECKPOINTS_DIR = os.path.join(REGRESSOR_DIR, "checkpoints")
CHECKPOINTS_FILE = os.path.join(CHECKPOINTS_DIR, REGRESSOR_TYPE+".ckpt")

print("\n")
print("Run infos:")
print("    REGRESSOR_TYPE: {}".format(REGRESSOR_TYPE))
print("    CHECKPOINTS_FILE: {}".format(CHECKPOINTS_FILE))
print("\n")
sys.stdout.flush()

if not os.path.exists(CHECKPOINTS_FILE):
    print("Checkpoint file not found. Please run 'python3 baseline_score_regressor.py -rt {}' to train score regressor".format(REGRESSOR_TYPE))
    sys.exit(-1)

print("Scoring labeled galaxy images using {}".format(REGRESSOR_TYPE))    

labels2paths = data.read_labels2paths(data_root="./data")
galaxy_images = labels2paths[1.0] # paths to galaxies

regr = joblib.load(CHECKPOINTS_FILE)

features, _,_ , ids = utils.features_summary(image_set=galaxy_images)
ids = ids.astype(int)

print("\n--Creating prediction.csv file for labeled galaxies--")
predictions = np.array(regr.predict(features))
predictions = np.clip(predictions, a_min=0, a_max=8).reshape([-1])
print("Predictions shape:", predictions.shape)
print("mean score: {}".format(np.mean(predictions)))
print("median score: {}".format(np.median(predictions)))

ids = ids.reshape([-1])
print("Ids shape:", ids.shape)

df = pd.DataFrame(data={'Id': ids, 'Score': predictions})

scores_file = os.path.join(".", "{}_predictions_labeled_galaxies.csv".format(REGRESSOR_TYPE))
df.to_csv(scores_file, index=False)
print("Saved predicted scores at {}".format(scores_file))

    

    

