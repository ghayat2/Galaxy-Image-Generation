import tensorflow as tf
import numpy as np
import sys, os, glob, gc
import pandas as pd
from data import create_dataloader_query
from Scorer import Scorer
from tqdm import trange
import datetime, time
from argparse import ArgumentParser

global_seed=5

tf.random.set_random_seed(global_seed)
np.random.seed(global_seed)

parser = ArgumentParser()
parser.add_argument('-bs', '--batch_size', type = int, default = 16, help = 'size of training batch')
parser.add_argument('-bp', '--batches_to_prefetch', type = int, default = 2, help = 'number of batches to prefetch')
parser.add_argument('-zc', '--zero_center', help = 'whether to zero_center image data i,e normalize to [-1, 1] or not i,e normalize to [0, 1]', action="store_true")

args = parser.parse_args()

def timestamp():
    return datetime.datetime.fromtimestamp(time.time()).strftime("%Y.%m.%d-%H:%M:%S")

def create_zip_code_files(output_file, submission_files):
    patoolib.create_archive(output_file, submission_files)

CURR_TIMESTAMP=timestamp()

BATCH_SIZE= 1 #args.batch_size
BATCHES_TO_PREFETCH=args.batches_to_prefetch
ZERO_CENTER = args.zero_center

C, H, W = 1, 1000, 1000 # images dimensions

# paths
DATA_ROOT="./data"
CLUSTER_DATA_ROOT="/cluster/scratch/mamrani/data"
if os.path.exists(CLUSTER_DATA_ROOT):
    DATA_ROOT=CLUSTER_DATA_ROOT

list_of_files = glob.glob('./LOG_SCORER/*')
latest_dir = max(list_of_files, key=os.path.getctime) # latest created dir for latest experiment
LOG_DIR=latest_dir
CHECKPOINTS_PATH = os.path.join(LOG_DIR, "checkpoints")
PREDICTIONS_DIR = os.path.join(LOG_DIR, "predictions", CURR_TIMESTAMP)

if not os.path.exists(PREDICTIONS_DIR):
    os.makedirs(PREDICTIONS_DIR)
    
# printing parameters
print("\n")
print("Run infos:")
print("    BATCH_SIZE: {}".format(BATCH_SIZE))
print("    BATCHES_TO_PREFETCH: {}".format(BATCHES_TO_PREFETCH))
print("    ZERO_CENTER: {}".format(ZERO_CENTER))
print("    DATA_ROOT: {}".format(DATA_ROOT))
print("    LOG_DIR: {}".format(LOG_DIR))
print("    PREDICTIONS_DIR: {}".format(PREDICTIONS_DIR))
print("\n")
sys.stdout.flush()

#sys.exit(0)
# remove warning messages
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
tf.logging.set_verbosity(tf.logging.ERROR)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0"
with tf.Session(config=config) as sess:
    
    # Scorer model
    test_im, all_paths, nb_images = create_dataloader_query(data_root=DATA_ROOT, batches_to_prefetch=BATCHES_TO_PREFETCH)

    if not ZERO_CENTER:
        test_im = (test_im+1.0)/2.0 # renormalize to [0, 1]
    
    print("Building Scorer model ...")
    sys.stdout.flush()
    model= Scorer()
    scores_pred = model(inp=test_im, training=False, zero_centered=ZERO_CENTER)
    
    images_ids = [path.split("/")[-1].split(".")[0] for path in all_paths]
    scores = np.empty([nb_images])
    
    print("Restoring latest model from {}\n".format(CHECKPOINTS_PATH))
    saver = tf.train.Saver()
    latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINTS_PATH)
    print("Latest checkpoint: {}".format(latest_checkpoint))
    saver.restore(sess, latest_checkpoint)

    with trange(nb_images) as t:
        for i in t: 
            score = sess.run(scores_pred)
            scores[i] = score[0,0]
    
    predictions = pd.DataFrame(data={'Id': images_ids, 'Predicted': scores})
#    print(predictions)
    pred_file = os.path.join(PREDICTIONS_DIR, "predictions.csv")
    predictions.to_csv(pred_file, index=False)
    print("Saved predictions at {}".format(pred_file))
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
