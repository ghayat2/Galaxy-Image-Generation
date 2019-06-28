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
parser.add_argument('-n_dim', '--noise_dim', type = int, default = 1000, help = 'the dimension of the noise input to the generator')
parser.add_argument('-to_gen', '--to_generate', type = int, default = 100, help = 'the number of samples to generate')
parser.add_argument('-ns', '--nb_stacks', type = int, default = 4, help = 'number of stacks')
parser.add_argument('-m', '--margin', type = float, default = 0.25, help = 'margin to add to the mean scores of training galaxies')

args = parser.parse_args()

def timestamp():
    return datetime.datetime.fromtimestamp(time.time()).strftime("%Y.%m.%d-%H:%M:%S")

CURR_TIMESTAMP=timestamp()

BATCH_SIZE=1 # generate images one by one

C, H, W = 1, 1000, 1000 # images dimensions
NOISE_DIM=args.noise_dim
TO_GENERATE = args.to_generate
NB_STACKS=args.nb_stacks
MARGIN = args.margin

# DCGAN paths
list_of_files = glob.glob('./LOG_DCGAN/*')
latest_dir = max(list_of_files, key=os.path.getctime) # latest created dir for latest experiment
LOG_DIR_DCGAN=latest_dir
CHECKPOINTS_PATH_DCGAN = os.path.join(LOG_DIR_DCGAN, "checkpoints")

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

LOG_DIR = os.path.join("./LOG_COMBINED", CURR_TIMESTAMP)
GENERATED_SAMPLES_DIR= os.path.join(LOG_DIR, "generated_samples")

class Logger(object):  # logger to log output to both terminal and file
    def __init__(self, log_dir):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        self.terminal = sys.stdout
        self.log = open(os.path.join(log_dir, "output"), "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        self.log.flush()
        self.terminal.flush()
        pass    

sys.stdout = Logger(LOG_DIR)

# printing parameters
print("\n")
print("Run infos:")
print("    BATCH_SIZE: {}".format(BATCH_SIZE))
print("    NOISE_DIM: {}".format(NOISE_DIM))
print("    LOG_DIR_DCGAN: {}".format(LOG_DIR_DCGAN))
print("    LOG_DIR_SRM: {}".format(LOG_DIR_SRM))
print("    LOG_DIR_SCORER: {}".format(LOG_DIR_SCORER))
print("    GENERATED_SAMPLES_DIR: {}".format(GENERATED_SAMPLES_DIR))
print("    TO_GENERATE: {}".format(TO_GENERATE))
print("    MARGIN: {}".format(MARGIN))
print("\n")
sys.stdout.flush()

#sys.exit(0)
# remove warning messages
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
tf.logging.set_verbosity(tf.logging.ERROR)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0"

gen_graph = tf.Graph()
gen_sess = tf.Session(graph=gen_graph, config=config)
with gen_graph.as_default():
    # DCGAN Generator model
    # define noise and test data
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM], seed=global_seed, name="random_noise") # noise fed to generator

    print("Building DCGAN Generator model ...")
    sys.stdout.flush()
    model= DCGAN()
    fake_im, _ = model.generator_model(noise=noise, training=False) # get fake images from generator

    print("Restoring latest model from {}".format(CHECKPOINTS_PATH_DCGAN))
    saver = tf.train.Saver()
    latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINTS_PATH_DCGAN)
    print("Latest checkpoint: {}\n".format(latest_checkpoint))
    saver.restore(gen_sess, latest_checkpoint)
    
#sys.exit(0)

srm_graph = tf.Graph()
srm_sess = tf.Session(graph=srm_graph, config=config)
with srm_graph.as_default():
    # StackedSRM model
    print("Building StackedSRM model ...")
    sys.stdout.flush()
    fake_im_pl = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, 1, 64, 64])
    model= StackedSRM(NB_STACKS)
    outputs_pred = model(inp=fake_im_pl, training=False)
    
    print("Restoring latest model from {}".format(CHECKPOINTS_PATH_SRM))
    saver = tf.train.Saver()
    latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINTS_PATH_SRM)
    print("Latest checkpoint: {}\n".format(latest_checkpoint))
    saver.restore(srm_sess, latest_checkpoint)
    
#sys.exit(0)

scorer_graph = tf.Graph()
scorer_sess = tf.Session(graph=scorer_graph, config=config)
with scorer_graph.as_default():

    # data
    galaxy_im, _, nb_galaxies, _ = create_dataloader_train_labeled(data_root=DATA_ROOT, batch_size=1, batches_to_prefetch=2, all_data=False)
    
    # DCGAN Scorer model
    print("Building DCGAN Scorer model ...")
    sys.stdout.flush()
    im_pl = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, 1, 1000, 1000])
    model1 = DCGAN()
    _, ops = model1.discriminator_model(inp=im_pl, training=False, resize=True) # get discriminator output

    flat = ops["flat"]
    model2 = Scorer_head()
    scores_pred = model2.scorer_head_model(features=flat, training=False)
    
    print("Restoring latest model from {}".format(CHECKPOINTS_PATH_SCORER))
    saver = tf.train.Saver()
    latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINTS_PATH_SCORER)
    print("Latest checkpoint: {}\n".format(latest_checkpoint))
    saver.restore(scorer_sess, latest_checkpoint)
    
#sys.exit(0)

print("Scoring training galaxy images")
galaxies_scores = []

with trange(nb_galaxies) as t:
    for i in t:
        im_val = scorer_sess.run(galaxy_im)
        score = scorer_sess.run(scores_pred, {im_pl: im_val})
        galaxies_scores.append(score[0,0])
    
galaxies_scores = np.array(galaxies_scores)
median_score = np.median(galaxies_scores)
mean_score = np.mean(galaxies_scores)
print("Median score: {}".format(median_score))
print("Mean score: {}".format(mean_score))

threshold = mean_score + MARGIN

print("Filtering images having score less than {}".format(threshold))
#sys.exit(0)

if not os.path.exists(GENERATED_SAMPLES_DIR):
    os.makedirs(GENERATED_SAMPLES_DIR)

counter = 0
generated_images_scores = []
generated_images_names = []
while counter < TO_GENERATE:
    print(counter)
    fake_im_val = gen_sess.run(fake_im)
#    print(fake_im_val.shape)
    fake_im_val = ((fake_im_val)+1)/2.0 # renormalize to [0, 1] to feed it to StackedSRM model
    
    srm_feed_dict = {fake_im_pl: fake_im_val}
    last_output = srm_sess.run(outputs_pred[-1], srm_feed_dict)[:, :, 12:-12, 12:-12] # get the last output of the StackedSRM model and remove padding (i,e convert to 1000x1000)
    
#    print(last_output.shape)

    scorer_input = (last_output*2.0)-1 # renormalize to [-1, 1] to feed to scorer model
    
    score = scorer_sess.run(scores_pred, {im_pl: scorer_input})[0, 0]
    
    if score < threshold:
        print("Filtering image with score", score)
        continue
    
    print("Keeping image with score {}", score)

    img = (last_output[0]*255.0).transpose(1,2,0).astype("uint8")[:, :, 0] # denormalize output and convert to channels last format
    #--------------------------------------------------------
#    max_val = img.max()
#    img = ((img/max_val)*255.0).astype("uint8")
    #-------------------------------------------------------- 
    
    print(img.shape)
    print("min: {}, max: {}".format(img.min(), img.max()))
    image = Image.fromarray(img)
    filename = "img_{}".format(counter)
    image.save(os.path.join(GENERATED_SAMPLES_DIR, filename+".png"))

    generated_images_scores.append(score)
    generated_images_names.append(filename)
    counter+=1
    
df = pd.DataFrame(data={'Id': generated_images_names, 'Score': generated_images_scores})

scores_file = os.path.join(LOG_DIR, "gen_images_scores.csv")
df.to_csv(scores_file, index=False)
print("Saved generated images scores at {}".format(scores_file))
    
    
    
    
    
    
    
    
