import tensorflow as tf
import numpy as np
import sys, os, glob, gc
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from CGAN import CGAN
from tqdm import trange
from PIL import Image
import datetime, time
from argparse import ArgumentParser

global_seed=5

tf.random.set_random_seed(global_seed)
np.random.seed(global_seed)

parser = ArgumentParser()
parser.add_argument('-bs', '--batch_size', type = int, default = 16, help = 'size of training batch')
parser.add_argument('-n_dim', '--noise_dim', type = int, default = 1000, help = 'the dimension of the noise input to the generator')
parser.add_argument('-to_gen', '--to_generate', type = int, default = 100, help = 'the number of samples to generate')

args = parser.parse_args()

def timestamp():
    return datetime.datetime.fromtimestamp(time.time()).strftime("%Y.%m.%d-%H:%M:%S")

CURR_TIMESTAMP=timestamp()

BATCH_SIZE=args.batch_size

C, H, W = 1, 1000, 1000 # images dimensions
NOISE_DIM=args.noise_dim
TO_GENERATE = args.to_generate

# paths
list_of_files = glob.glob('./LOG_CGAN/*')
latest_dir = max(list_of_files, key=os.path.getctime) # latest created dir for latest experiment
LOG_DIR=latest_dir
CHECKPOINTS_PATH = os.path.join(LOG_DIR, "checkpoints")
GENERATED_SAMPLES_DIR = os.path.join(LOG_DIR, "generated_samples")

# printing parameters
print("\n")
print("Run infos:")
print("    BATCH_SIZE: {}".format(BATCH_SIZE))
print("    NOISE_DIM: {}".format(NOISE_DIM))
print("    LOG_DIR: {}".format(LOG_DIR))
print("    CHECKPOINTS_PATH: {}".format(CHECKPOINTS_PATH))
print("    GENERATED_SAMPLES_DIR: {}".format(GENERATED_SAMPLES_DIR))
print("    TO_GENERATE: {}".format(TO_GENERATE))
print("\n")
sys.stdout.flush()

# remove warning messages
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
tf.logging.set_verbosity(tf.logging.ERROR)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0"
with tf.Session(config=config) as sess:
    
    # define noise and test data
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM], seed=global_seed, name="random_noise") # noise fed to generator
    y_G = tf.ones([BATCH_SIZE, 1]) # labels fed to generator, always 1
    
    # define placeholders
    noise_pl = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, NOISE_DIM]) # placeholder for noise fed to generator
    y_pl_G = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, 1]) # placeholder for label fed to generator
    
    training_pl = tf.placeholder(dtype=tf.bool, shape=[])
    
    #model
    print("Building model ...")
    sys.stdout.flush()
    model= CGAN()
    fake_im, _ = model.generator_model(noise=noise_pl, y=y_pl_G, training=training_pl) # get fake images from generator
    
    print("Restoring latest model from {}\n".format(CHECKPOINTS_PATH))
    saver = tf.train.Saver()
    saver.restore(sess,tf.train.latest_checkpoint(CHECKPOINTS_PATH))
    
#    sys.exit(0)
    
    if not os.path.exists(GENERATED_SAMPLES_DIR):
        os.makedirs(GENERATED_SAMPLES_DIR)
        
    counter = 0
    while counter < TO_GENERATE:
        noise_val, y_val_G = sess.run([noise, y_G])
        feed_dict={training_pl:False, noise_pl: noise_val, y_pl_G: y_val_G} # feed dict
        
        images = sess.run(fake_im, feed_dict)
        
        for image in images:
            print(counter)
            image = ((image+1)*128.0).transpose(1,2,0).astype("uint8")[:, :, 0] # unnormalize image and put channels_last and remove the channels dimension
            image = Image.fromarray(image)
            image.save(os.path.join(GENERATED_SAMPLES_DIR, "img_{}.png".format(counter)))
            counter+=1
            if counter >= TO_GENERATE:
                break
    
    
    
    
    
    
    
    
    
    
