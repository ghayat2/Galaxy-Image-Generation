import tensorflow as tf
import numpy as np
import sys, os, glob, gc
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from data import create_dataloader_train
from DCGAN import DCGAN
from StackedSRM import StackedSRM
from tqdm import trange
from PIL import Image
import datetime, time
from argparse import ArgumentParser

global_seed=5

tf.random.set_random_seed(global_seed)
np.random.seed(global_seed)

parser = ArgumentParser()
parser.add_argument('-n_dim', '--noise_dim', type = int, default = 1000, help = 'the dimension of the noise input to the generator')
parser.add_argument('-to_gen', '--to_generate', type = int, default = 100, help = 'the number of samples to generate')
parser.add_argument('-ns', '--nb_stacks', type = int, default = 3, help = 'number of stacks')

args = parser.parse_args()

def timestamp():
    return datetime.datetime.fromtimestamp(time.time()).strftime("%Y.%m.%d-%H:%M:%S")

CURR_TIMESTAMP=timestamp()

BATCH_SIZE=1 # generate images one by one

C, H, W = 1, 1000, 1000 # images dimensions
NOISE_DIM=args.noise_dim
TO_GENERATE = args.to_generate
NB_STACKS=args.nb_stacks

# DCGAN paths
list_of_files = glob.glob('./LOG_DCGAN/*')
latest_dir = max(list_of_files, key=os.path.getctime) # latest created dir for latest experiment
LOG_DIR_DCGAN=latest_dir
CHECKPOINTS_PATH_DCGAN = os.path.join(LOG_DIR_DCGAN, "checkpoints")
GENERATED_SAMPLES_DIR_DCGAN = os.path.join(LOG_DIR_DCGAN, "generated_samples")

# StackedSRM paths
list_of_files = glob.glob('./LOG_SRM/*')
latest_dir = max(list_of_files, key=os.path.getctime) # latest created dir for latest experiment
LOG_DIR_SRM=latest_dir
CHECKPOINTS_PATH_SRM = os.path.join(LOG_DIR_SRM, "checkpoints")
GENERATED_SAMPLES_DIR= os.path.join("./LOG_COMBINED", CURR_TIMESTAMP, "generated_samples")

# printing parameters
print("\n")
print("Run infos:")
print("    BATCH_SIZE: {}".format(BATCH_SIZE))
print("    NOISE_DIM: {}".format(NOISE_DIM))
print("    LOG_DIR_DCGAN: {}".format(LOG_DIR_DCGAN))
print("    LOG_DIR_SRM: {}".format(LOG_DIR_SRM))
print("    GENERATED_SAMPLES_DIR: {}".format(GENERATED_SAMPLES_DIR))
print("    TO_GENERATE: {}".format(TO_GENERATE))
print("\n")
sys.stdout.flush()

#sys.exit(0)
# remove warning messages
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
tf.logging.set_verbosity(tf.logging.ERROR)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0"

dcgan_graph = tf.Graph()
dcgan_sess = tf.Session(graph=dcgan_graph, config=config)
with dcgan_graph.as_default():
    # DCGAN model
    # define noise and test data
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM], name="random_noise") # noise fed to generator

    print("Building DCGAN model ...")
    sys.stdout.flush()
    model= DCGAN()
    fake_im, _ = model.generator_model(noise=noise, training=False) # get fake images from generator

    print("Restoring latest model from {}\n".format(CHECKPOINTS_PATH_DCGAN))
    saver = tf.train.Saver()
    saver.restore(dcgan_sess, tf.train.latest_checkpoint(CHECKPOINTS_PATH_DCGAN))
    
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
    
    print("Restoring latest model from {}\n".format(CHECKPOINTS_PATH_SRM))
    saver = tf.train.Saver()
    saver.restore(srm_sess, tf.train.latest_checkpoint(CHECKPOINTS_PATH_SRM))
    
#sys.exit(0)

if not os.path.exists(GENERATED_SAMPLES_DIR):
    os.makedirs(GENERATED_SAMPLES_DIR)

counter = 0
while counter < TO_GENERATE:
    print(counter)
    fake_im_val = dcgan_sess.run(fake_im)
#    print(fake_im_val.shape)
    fake_im_val = ((fake_im_val)+1)/2.0 # renormalize to [0, 1] to feed it to StackedSRM model
    
    srm_feed_dict = {fake_im_pl: fake_im_val}
    last_output = srm_sess.run(outputs_pred[-1], srm_feed_dict) # get the last output of the StackedSRM model
    
#    print(last_output.shape)

    img = (last_output[0]*255.0).transpose(1,2,0).astype("uint8")[:, :, 0] # denormalize output and convert to channels last format
    
    print(img.shape)
    print("min: {}, max: {}".format(img.min(), img.max()))
    image = Image.fromarray(img)
    image.save(os.path.join(GENERATED_SAMPLES_DIR, "img_{}.png".format(counter)))
    
    counter+=1

    
    
    
    
    
    
    
    
