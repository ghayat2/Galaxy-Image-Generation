import numpy as np
import sys, os, glob
import pandas as pd
import utils
from sklearn.externals import joblib
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-d', '--images_dir', type = str, required=True, help = 'path to directory with images 1000x1000')
parser.add_argument('-rt', '--regressor_type', type = str, default = "Boost", choices=["Random_Forest", "Ridge", "Boost"], 
                        help = 'type of regressor trained on the manually extracted features and used to make predictions on the query image dataset')
                        
args = parser.parse_args()

REGRESSOR_TYPE = args.regressor_type
IMAGES_DIR = args.images_dir

REGRESSOR_DIR = os.path.join("./Regressor", REGRESSOR_TYPE)
CHECKPOINTS_DIR = os.path.join(REGRESSOR_DIR, "checkpoints")
CHECKPOINTS_FILE = os.path.join(CHECKPOINTS_DIR, REGRESSOR_TYPE+".ckpt")
SAVE_DIR = os.path.dirname(IMAGES_DIR.rstrip("/")) # save at the parent directory of IMAGES_DIR

print("\n")
print("Run infos:")
print("    REGRESSOR_TYPE: {}".format(REGRESSOR_TYPE))
print("    CHECKPOINTS_FILE: {}".format(CHECKPOINTS_FILE))
print("    IMAGES_DIR: {}".format(IMAGES_DIR))
print("    SAVE_DIR: {}".format(SAVE_DIR))
print("\n")
sys.stdout.flush()

if not os.path.exists(IMAGES_DIR):
    print("IMAGES DIR: {} not found".format(IMAGES_DIR))
    sys.exit(-1)

if not os.path.exists(CHECKPOINTS_FILE):
    print("Checkpoint file not found. Please run 'python3 baseline_score_regressor.py -rt {}' to train score regressor".format(REGRESSOR_TYPE))
    sys.exit(-1)
    
if len(glob.glob(os.path.join(IMAGES_DIR, "*"))) == 0:
    print("Empty images directory.")
    sys.exit(0)

print("Extracting features for images...")
features, _, _, ids = utils.extract_features(image_dir=IMAGES_DIR)

regr = joblib.load(CHECKPOINTS_FILE)

#sys.exit(0)

print("\n--Creating prediction.csv file--")
predictions = np.array(regr.predict(features))
predictions = np.clip(predictions, a_min=0, a_max=8).reshape([-1])
print("Predictions shape:", predictions.shape)
print("mean score: {}".format(np.mean(predictions)))
print("median score: {}".format(np.median(predictions)))

ids = ids.reshape([-1])
print("Ids shape:", ids.shape)

df = pd.DataFrame(data={'Id': ids, 'Score': predictions})

scores_file = os.path.join(SAVE_DIR, "{}_predictions.csv".format(REGRESSOR_TYPE))
df.to_csv(scores_file, index=False)
print("Saved predicted scores at {}".format(scores_file))

