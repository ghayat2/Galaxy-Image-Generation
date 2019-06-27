import tensorflow as tf
import numpy as np
import sys, os, glob, gc
import pandas as pd
from data import create_dataloader_train_labeled
from DCGAN import DCGAN
from StackedSRM import StackedSRM
from DCGAN_Scorer import Scorer_head
from tqdm import trange
from PIL import Image
import datetime, time
from argparse import ArgumentParser
import layers

global_seed=5

tf.random.set_random_seed(global_seed)
np.random.seed(global_seed)

parser = ArgumentParser()
parser.add_argument('-d', '--images_dir', type = str, required=True, help = 'path to directory with images 64x64 to upsample')
parser.add_argument('-ns', '--nb_stacks', type = int, default = 4, help = 'number of stacks')

args = parser.parse_args()

def timestamp():
    return datetime.datetime.fromtimestamp(time.time()).strftime("%Y.%m.%d-%H:%M:%S")

CURR_TIMESTAMP=timestamp()

BATCH_SIZE=1 # generate images one by one

C, H, W = 1, 1000, 1000 # images dimensions
NB_STACKS=args.nb_stacks
INPUT_IMAGES_DIR=args.images_dir

# StackedSRM paths
list_of_files = glob.glob('./LOG_SRM/*')
latest_dir = max(list_of_files, key=os.path.getctime) # latest created dir for latest experiment
LOG_DIR_SRM=latest_dir
CHECKPOINTS_PATH_SRM = os.path.join(LOG_DIR_SRM, "checkpoints")

# Scorer paths
list_of_files = glob.glob('./LOG_DCGAN_SCORER/*')
latest_dir = max(list_of_files, key=os.path.getctime) # latest created dir for latest experiment
LOG_DIR_SCORER=latest_dir
CHECKPOINTS_PATH_SCORER = os.path.join(LOG_DIR_SCORER, "checkpoints")

DATA_ROOT="./data"
CLUSTER_DATA_ROOT="/cluster/scratch/mamrani/data"
if os.path.exists(CLUSTER_DATA_ROOT):
    DATA_ROOT=CLUSTER_DATA_ROOT

LOG_DIR = os.path.join("./LOG_UPSAMPLED", CURR_TIMESTAMP)
GENERATED_SAMPLES_DIR= os.path.join(LOG_DIR, "upsampled_samples")

# printing parameters
print("\n")
print("Run infos:")
print("    BATCH_SIZE: {}".format(BATCH_SIZE))
print("    LOG_DIR_SRM: {}".format(LOG_DIR_SRM))
print("    LOG_DIR_SCORER: {}".format(LOG_DIR_SCORER))
print("    INPUT_IMAGES_DIR: {}".format(INPUT_IMAGES_DIR))
print("    GENERATED_SAMPLES_DIR: {}".format(GENERATED_SAMPLES_DIR))
print("\n")
sys.stdout.flush()

images_paths = glob.glob(os.path.join(INPUT_IMAGES_DIR, "*"))
#print(images_paths)

#sys.exit(0)
# remove warning messages
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
tf.logging.set_verbosity(tf.logging.ERROR)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0"

srm_graph = tf.Graph()
srm_sess = tf.Session(graph=srm_graph, config=config)
with srm_graph.as_default():
    # StackedSRM model
    print("Building StackedSRM model ...")
    sys.stdout.flush()
    srm_im_pl = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, 1, 64, 64])
    model= StackedSRM(NB_STACKS)
    outputs_pred = model(inp=srm_im_pl, training=False)
    
    print("Restoring latest model from {}".format(CHECKPOINTS_PATH_SRM))
    saver = tf.train.Saver()
    latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINTS_PATH_SRM)
    print("Latest checkpoint: {}\n".format(latest_checkpoint))
    saver.restore(srm_sess, latest_checkpoint)
    
#sys.exit(0)

scorer_graph = tf.Graph()
scorer_sess = tf.Session(graph=scorer_graph, config=config)
with scorer_graph.as_default():

    # DCGAN Scorer model
    print("Building DCGAN Scorer model ...")
    sys.stdout.flush()
    scorer_im_pl = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, 1, 1000, 1000])
    model1 = DCGAN()
    _, ops = model1.discriminator_model(inp=scorer_im_pl, training=False, resize=True) # get discriminator output

    flat = ops["flat"]
    model2 = Scorer_head()
    scores_pred = model2.scorer_head_model(features=flat, training=False)
    
    print("Restoring latest model from {}".format(CHECKPOINTS_PATH_SCORER))
    saver = tf.train.Saver()
    latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINTS_PATH_SCORER)
    print("Latest checkpoint: {}\n".format(latest_checkpoint))
    saver.restore(scorer_sess, latest_checkpoint)
    
#sys.exit(0)

if not os.path.exists(GENERATED_SAMPLES_DIR):
    os.makedirs(GENERATED_SAMPLES_DIR)

upsampled_images_scores = []
upsampled_images_names = []
counter = 0
for path in images_paths:
    print(counter)
    im_val = np.array([np.array(Image.open(path)).reshape([64, 64, 1]).transpose(2,0,1)]) # read image and put in channels first
#    print(im_val.shape)

    im_val = (im_val)/255.0 # normalize to [0, 1] for feeding to SRM
    
    srm_feed_dict = {srm_im_pl: im_val}
    last_output = srm_sess.run(outputs_pred[-1], srm_feed_dict)[:, :, 12:-12, 12:-12] # get the last output of the StackedSRM model and remove padding
    
#    print(last_output.shape)

    scorer_input = (last_output*2.0)-1 # renormalize to [-1, 1] to feed to scorer model
    
    score = scorer_sess.run(scores_pred, {scorer_im_pl: scorer_input})[0, 0]

    img = (last_output[0]*255.0).transpose(1,2,0).astype("uint8")[:, :, 0] # denormalize output and convert to channels last format 
    
    print(img.shape)
    print("min: {}, max: {}".format(img.min(), img.max()))
    image = Image.fromarray(img)
    filename = path.split("/")[-1].split(".")[0]
#    print(filename)
    image.save(os.path.join(GENERATED_SAMPLES_DIR, filename+".png"))

    upsampled_images_scores.append(score)
    upsampled_images_names.append(filename)
    counter += 1
    
df = pd.DataFrame(data={'Id': upsampled_images_names, 'Score': upsampled_images_scores})

scores_file = os.path.join(LOG_DIR, "gen_images_scores.csv")
df.to_csv(scores_file, index=False)
print("Saved generated images scores at {}".format(scores_file))
    
    
    
    
    
    
    
    
