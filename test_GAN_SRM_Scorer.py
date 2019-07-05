import tensorflow as tf
import numpy as np
import sys, os, glob, gc
import pandas as pd
from data import create_dataloader_train_labeled, read_labels2paths
from DCGAN import DCGAN
from MCGAN import MCGAN
from FullresGAN import FullresGAN
from StackedSRM import StackedSRM
from DCGAN_Scorer import Scorer_head
from tqdm import trange
from PIL import Image
import datetime, time
import utils
from sklearn.externals import joblib
from argparse import ArgumentParser
import layers
from tools import *

global_seed=5

tf.random.set_random_seed(global_seed)
np.random.seed(global_seed)

parser = ArgumentParser()
parser.add_argument('-n_dim', '--noise_dim', type = int, default = 1000, help = 'the dimension of the noise input to the generator')
parser.add_argument('-to_gen', '--to_generate', type = int, default = 100, help = 'the number of samples to generate')
parser.add_argument('-ns', '--nb_stacks', type = int, default = 4, help = 'number of stacks')
parser.add_argument('-gen', '--generator', type = str, default = "DCGAN", choices=["DCGAN", "MCGAN", "FullresGAN"], help = 'the generator used to generate galaxy images')
parser.add_argument('-sc', '--scorer', type = str, default = None, choices=["DCGAN_Scorer", "Random_Forest", "Ridge", "Boost"], help = 'the regressor used to score the generated images')
parser.add_argument('-m', '--margin', type = float, default = 0.25, help = 'margin to add to the mean score of training galaxies')
parser.add_argument('-t', '--threshold', type = float, default = 3.0, help = 'threshold score for generated galaxies')
parser.add_argument('-ut', '--use_threshold', help = 'whether to use the scorer with a threshold to filter the generated images by score', action="store_true")
parser.add_argument('-um', '--use_margin', help = 'whether to use the scorer with a margin on the mean score of training galaxies to filter the generated images by score', action="store_true")

args = parser.parse_args()

CURR_TIMESTAMP=timestamp()

BATCH_SIZE=1 # generate images one by one

C, H, W = 1, 1000, 1000 # images dimensions
NOISE_DIM=args.noise_dim
FEATS_DIM=1
TO_GENERATE = args.to_generate
NB_STACKS=args.nb_stacks
GENERATOR = args.generator
MARGIN = args.margin
THRESHOLD=args.threshold
USE_SCORER_THRES = args.use_threshold
USE_SCORER_MARGIN = args.use_margin
SCORER = args.scorer
if USE_SCORER_THRES and USE_SCORER_MARGIN: # if both set, prefer threshold scoring
    USE_SCORER_MARGIN=False
USE_SCORER=USE_SCORER_THRES or USE_SCORER_MARGIN

# DCGAN paths
if GENERATOR == "DCGAN":
    list_of_files = glob.glob('./LOG_DCGAN/*')
    latest_dir = max(list_of_files, key=os.path.getctime) # latest created dir for latest experiment
    LOG_DIR_DCGAN=latest_dir
    CHECKPOINTS_PATH_DCGAN = os.path.join(LOG_DIR_DCGAN, "checkpoints")
else:
    LOG_DIR_DCGAN=None

# MCGAN paths
if GENERATOR == "MCGAN":
    list_of_files = glob.glob('./LOG_MCGAN/*')
    latest_dir = max(list_of_files, key=os.path.getctime) # latest created dir for latest experiment
    LOG_DIR_MCGAN=latest_dir
    CHECKPOINTS_PATH_MCGAN = os.path.join(LOG_DIR_MCGAN, "checkpoints")
else:
    LOG_DIR_MCGAN=None
 
# Fullres paths
if GENERATOR == "FullresGAN":
    list_of_files = glob.glob('./LOG_FullresGAN/*')
    latest_dir = max(list_of_files, key=os.path.getctime) # latest created dir for latest experiment
    LOG_DIR_FullresGAN=latest_dir
    CHECKPOINTS_PATH_FullresGAN = os.path.join(LOG_DIR_FullresGAN, "checkpoints")
else:
    LOG_DIR_FullresGAN=None

# StackedSRM paths
list_of_files = glob.glob('./LOG_SRM/*')
latest_dir = max(list_of_files, key=os.path.getctime) # latest created dir for latest experiment
LOG_DIR_SRM=latest_dir
CHECKPOINTS_PATH_SRM = os.path.join(LOG_DIR_SRM, "checkpoints")

# Scorer paths
if SCORER == "DCGAN_Scorer":
    list_of_files = glob.glob('./LOG_DCGAN_SCORER/*')
    latest_dir = max(list_of_files, key=os.path.getctime) # latest created dir for latest experiment
    LOG_DIR_DCGAN_SCORER=latest_dir
    CHECKPOINTS_PATH_SCORER = os.path.join(LOG_DIR_DCGAN_SCORER, "checkpoints")
else:
    LOG_DIR_DCGAN_SCORER=None

DATA_ROOT="./data"

LOG_DIR = os.path.join("./LOG_COMBINED", CURR_TIMESTAMP)
GENERATED_SAMPLES_DIR= os.path.join(LOG_DIR, "generated_samples")

sys.stdout = Logger(LOG_DIR)

# printing parameters
print("\n")
print("Run infos:")
print("    GENERATOR: {}".format(GENERATOR))
print("    BATCH_SIZE: {}".format(BATCH_SIZE))
print("    NOISE_DIM: {}".format(NOISE_DIM))
print("    LOG_DIR_DCGAN: {}".format(LOG_DIR_DCGAN))
print("    LOG_DIR_MCCGAN: {}".format(LOG_DIR_MCGAN))
print("    LOG_DIR_FullresGAN: {}".format(LOG_DIR_FullresGAN))
print("    LOG_DIR_SRM: {}".format(LOG_DIR_SRM))
print("    LOG_DIR_DCGAN_SCORER: {}".format(LOG_DIR_DCGAN_SCORER))
print("    GENERATED_SAMPLES_DIR: {}".format(GENERATED_SAMPLES_DIR))
print("    TO_GENERATE: {}".format(TO_GENERATE))
print("    MARGIN: {}".format(MARGIN))
print("    THRESHOLD: {}".format(THRESHOLD))
print("    SCORER: {}".format(SCORER))
print("    USE_SCORER: {}".format(USE_SCORER))
print("    USE_SCORER_THRES: {}".format(USE_SCORER_THRES))
print("    USE_SCORER_MARGIN: {}".format(USE_SCORER_MARGIN))
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
    if GENERATOR == "DCGAN":
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
    
    elif GENERATOR == "MCGAN":
        # DCGAN Generator model
        # define noise and test data
        noise = tf.random.normal([BATCH_SIZE, NOISE_DIM], seed=global_seed, name="random_noise") # noise fed to generator
        feats = tf.cast(tf.random.uniform([BATCH_SIZE, FEATS_DIM], minval=1, maxval=20, dtype=tf.int32, seed=global_seed), tf.float32)
        
        print("Building MCGAN Generator model ...")
        sys.stdout.flush()
        model= MCGAN()
        fake_im, _ = model.generator_model(noise=noise, feats=feats, training=False) # get fake images from generator
        
        print("Restoring latest model from {}".format(CHECKPOINTS_PATH_MCGAN))
        saver = tf.train.Saver()
        latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINTS_PATH_MCGAN)
        print("Latest checkpoint: {}\n".format(latest_checkpoint))
        saver.restore(gen_sess, latest_checkpoint)

    elif GENERATOR == "FullresGAN":
        # DCGAN Generator model
        # define noise and test data
        noise = tf.random.normal([BATCH_SIZE, NOISE_DIM], seed=global_seed, name="random_noise") # noise fed to generator

        print("Building FullresGAN Generator model ...")
        sys.stdout.flush()
        model= FullresGAN()
        fake_im, _ = model.generator_model(noise=noise, training=False) # get fake images from generator

        print("Restoring latest model from {}".format(CHECKPOINTS_PATH_FullresGAN))
        saver = tf.train.Saver()
        latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINTS_PATH_FullresGAN)
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

if USE_SCORER and SCORER == "DCGAN_Scorer":
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

#    sys.exit(0)
    if USE_SCORER_MARGIN:
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
    else:
        threshold = THRESHOLD

    print("Filtering images having score less than {}".format(threshold))
    
elif USE_SCORER: # using a baseline regressor to score images
    labels2paths = read_labels2paths(data_root=DATA_ROOT)
    galaxies_images_paths = labels2paths[1.0] # paths to galaxies
    
    CHECKPOINTS_FILE = os.path.join("./Regressor", SCORER, "checkpoints", SCORER+".ckpt")
    print("{} checkpoint file: {}".format(SCORER, CHECKPOINTS_FILE))
    if not os.path.exists(CHECKPOINTS_FILE):
        print("Checkpoint file for scorer {} not found. Please run 'python3 baseline_score_regressor.py -rt {}' to train score regressor".format(SCORER, SCORER))
        sys.exit(-1)
    
    print("Loading {} model ...".format(SCORER))
    regr = joblib.load(CHECKPOINTS_FILE)

    if USE_SCORER_MARGIN:
        print("\nReading scores for training galaxy images")
        FEATURES_DIR=os.path.join("./data", "features")
        LABELED_FEATS_PATH = os.path.join(FEATURES_DIR, 'labeled_feats.gz')
        LABELED_FEATS_IDS_PATH = os.path.join(FEATURES_DIR, 'labeled_feats_ids.gz')
        
        if not os.path.exists(LABELED_FEATS_PATH) or not os.path.exists(LABELED_FEATS_IDS_PATH):
            print("Labeled dataset features '{}' or ids '{}' not found, please run 'python3 generate_feats.py -no_s -no_q' to generate features for the labeled dataset ...".format(LABELED_FEATS_PATH, LABELED_FEATS_IDS_PATH))
            sys.exit(-1)
        
        # read labeled dataset features
        labeled_feats = np.loadtxt(LABELED_FEATS_PATH)
        labeled_ids = np.loadtxt(LABELED_FEATS_IDS_PATH).astype(int)
        labeled_dict = dict(zip(labeled_ids, labeled_feats))
        
        galaxies_ids = [int(path.split("/")[-1].split(".")[0]) for path in galaxies_images_paths]
        
        features = [labeled_dict[id] for id in galaxies_ids]
        predictions = np.array(regr.predict(features))
        
        median_score = np.median(predictions)
        mean_score = np.mean(predictions)
        print("Median score: {}".format(median_score))
        print("Mean score: {}".format(mean_score))
        
        threshold = mean_score + MARGIN
    else:
        threshold = THRESHOLD

    print("Filtering images having score less than {}".format(threshold))

#sys.exit(0)

GENERATED_SAMPLES_64_DIR = os.path.join(GENERATED_SAMPLES_DIR, "64")
if not os.path.exists(GENERATED_SAMPLES_64_DIR):
    os.makedirs(GENERATED_SAMPLES_64_DIR)

GENERATED_SAMPLES_1000_DIR = os.path.join(GENERATED_SAMPLES_DIR, "1000")
if not os.path.exists(GENERATED_SAMPLES_1000_DIR):
    os.makedirs(GENERATED_SAMPLES_1000_DIR)

counter = 0
generated_images_scores = []
generated_images_names = []
while counter < TO_GENERATE:
    print(counter)
    fake_im_val = gen_sess.run(fake_im)
    fake_im_val = ((fake_im_val)+1)/2.0 # renormalize to [0, 1] to feed it to StackedSRM model
    
    if(GENERATOR != "FullresGAN"):
        srm_feed_dict = {fake_im_pl: fake_im_val}
        last_output = srm_sess.run(outputs_pred[-1], srm_feed_dict)[:, :, 12:-12, 12:-12] # get the last output of the StackedSRM model and remove padding (i,e convert to 1000x1000)
    else:
        last_output = fake_im_val

    img_64 = (fake_im_val[0]*255.0).transpose(1,2,0).astype("uint8")[:, :, 0] # denormalize output and convert to channels last format
    img_1000 = (last_output[0]*255.0).transpose(1,2,0).astype("uint8")[:, :, 0] # denormalize output and convert to channels last format
    
    if USE_SCORER:
        if SCORER == "DCGAN_Scorer":
            scorer_input = (np.array([img_1000]).reshape([1, 1000, 1000, 1]).transpose(0, 3, 1, 2))/128.0 - 1 # add batch_dim and channels_dim, put to channels last and renormalize to [-1, 1] 
                                                                                                              # to feed to scorer model
            
            score = scorer_sess.run(scores_pred, {im_pl: scorer_input})[0, 0]
        else:
            img = img_1000/255.0 # convert to channels last format and keep normalized to [0, 1]

            a = time.time()
            features = utils.get_hand_crafted(img).reshape([1, -1])
            print("time to extract feats: {}s".format(time.time()-a))

            a = time.time()
            score = np.array(regr.predict(features))[0]
            print("time to score: {}s".format(time.time()-a))
        
        if score < threshold:
            print("Filtering image with score", score)
            continue
        
        print("Keeping image with score {}", score)

    
    if(GENERATOR != "FullresGAN"):
        print("Saving with size 64x64")
        print(img_64.shape)
        print("min: {}, max: {}".format(img_64.min(), img_64.max()))
        image = Image.fromarray(img_64)
        filename = "img_{}".format(counter)
        image.save(os.path.join(GENERATED_SAMPLES_64_DIR, filename+".png"))
    
    print("Saving with size 1000x1000")
    print(img_1000.shape)
    print("min: {}, max: {}".format(img_1000.min(), img_1000.max()))
    image = Image.fromarray(img_1000)
    filename = "img_{}".format(counter)
    image.save(os.path.join(GENERATED_SAMPLES_1000_DIR, filename+".png"))
    
    if USE_SCORER:
        generated_images_scores.append(score)
        generated_images_names.append(filename)
    counter+=1

if USE_SCORER:
    df = pd.DataFrame(data={'Id': generated_images_names, 'Score': generated_images_scores})

    scores_file = os.path.join(LOG_DIR, "gen_images_scores.csv")
    df.to_csv(scores_file, index=False)
    print("Saved generated images scores at {}".format(scores_file))
    
    
    
    
    
    
    
    
