import tensorflow as tf
import numpy as np
import sys, os, glob, gc
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from data import create_dataloader_train_scored
from Scorer import Scorer
from tqdm import trange
from PIL import Image
import datetime, time
from argparse import ArgumentParser
import patoolib

global_seed=5

tf.random.set_random_seed(global_seed)
np.random.seed(global_seed)

parser = ArgumentParser()
parser.add_argument('-ne', '--num_epochs', type = int, default = 200, help = 'number of training epochs')
parser.add_argument('-bs', '--batch_size', type = int, default = 16, help = 'size of training batch')
parser.add_argument('-lr', '--learning_rate', type = float, default = 1e-3, help = 'learning rate for the optimizer')
parser.add_argument('-b1', '--beta_1', type = float, default = 0.5, help = 'beta 1 for the optimizer')
parser.add_argument('-b2', '--beta_2', type = float, default = 0.999, help = 'beta 2 for the optimizer')
parser.add_argument('-vp', '--valid_precent', type = float, default = 0.1, help = 'percentage of the data to use for validation')

parser.add_argument('-lf', '--log_iter_freq', type = int, default = 100, help = 'number of iterations between training logs')
parser.add_argument('-vdf', '--valid_iter_freq', type = int, default = 250, help = 'number of iterations between validation steps')
parser.add_argument('-svf', '--save_iter_freq', type = int, default = 2000, help = 'number of iterations between saving model checkpoints')

parser.add_argument('-bp', '--batches_to_prefetch', type = int, default = 2, help = 'number of batches to prefetch')
parser.add_argument('-ct', '--continue_training', help = 'whether to continue training from the last checkpoint of the last experiment or not', action="store_true")
parser.add_argument('-zc', '--zero_center', help = 'whether to zero_center image data i,e normalize to [-1, 1] or not i,e normalize to [0, 1]', action="store_true")
#parser.add_argument('-c', '--colab', help = 'whether we are running on colab or not', action="store_true") # add this option to specify that the code is run on colab

args = parser.parse_args()

def timestamp():
    return datetime.datetime.fromtimestamp(time.time()).strftime("%Y.%m.%d-%H:%M:%S")

def create_zip_code_files(output_file, submission_files):
    patoolib.create_archive(output_file, submission_files)

CURR_TIMESTAMP=timestamp()

NUM_EPOCHS=args.num_epochs
BATCH_SIZE=args.batch_size
BATCHES_TO_PREFETCH=args.batches_to_prefetch
LR = args.learning_rate # learning rate
BETA1 = args.beta_1
BETA2 = args.beta_2
VALID_PERCENT = args.valid_precent
LOG_ITER_FREQ = args.log_iter_freq # train loss logging frequency (in nb of steps)
SAVE_ITER_FREQ = args.save_iter_freq
VALID_ITER_FREQ = args.valid_iter_freq
CONTINUE_TRAINING = args.continue_training
ZERO_CENTER = args.zero_center

C, H, W = 1, 1000, 1000 # images dimensions
#RUNNING_ON_COLAB = args.colab

# paths
DATA_ROOT="./data"
CLUSTER_DATA_ROOT="/cluster/scratch/mamrani/data"
if os.path.exists(CLUSTER_DATA_ROOT):
    DATA_ROOT=CLUSTER_DATA_ROOT
LOG_DIR=os.path.join(".", "LOG_SCORER", CURR_TIMESTAMP)
if CONTINUE_TRAINING: # continue training from last training experiment
    list_of_files = glob.glob(os.path.join(".", "LOG_SCORER", "*"))
    LOG_DIR = max(list_of_files, key=os.path.getctime) # latest created dir for latest experiment will be our log path
CHECKPOINTS_PATH = os.path.join(LOG_DIR, "checkpoints")
SAMPLES_DIR = os.path.join(LOG_DIR, "test_samples")

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
print("    NUM_EPOCHS: {}".format(NUM_EPOCHS))
print("    BATCH_SIZE: {}".format(BATCH_SIZE))
print("    BETA1: {}".format(BETA1))
print("    BETA2: {}".format(BETA2))
print("    LEARNING_RATE: {}".format(LR))
print("    ZERO_CENTER: {}".format(ZERO_CENTER))
print("    BATCHES_TO_PREFETCH: {}".format(BATCHES_TO_PREFETCH))
print("    VALID_PERCENT: {}".format(VALID_PERCENT))
print("    LOG_ITER_FREQ: {}".format(LOG_ITER_FREQ))
print("    SAVE_ITER_FREQ: {}".format(SAVE_ITER_FREQ))
print("    VALID_ITER_FREQ: {}".format(VALID_ITER_FREQ))
print("    DATA_ROOT: {}".format(DATA_ROOT))
print("    LOG_DIR: {}".format(LOG_DIR))
print("    CONTINUE_TRAINING: {}".format(CONTINUE_TRAINING))
print("\n")
sys.stdout.flush()

files = ["data.py",
         "layers.py",
         "Scorer.py",
         "train_scorer.py"
         ]
         
if not CONTINUE_TRAINING:
    create_zip_code_files(os.path.join(LOG_DIR, "code.zip"), files)

#sys.exit(0)

# remove warning messages
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
tf.logging.set_verbosity(tf.logging.ERROR)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0"
with tf.Session(config=config) as sess:

    # data
    train_ds, nb_train, valid_ds, nb_valid = create_dataloader_train_scored(DATA_ROOT, batch_size=BATCH_SIZE, batches_to_prefetch=BATCHES_TO_PREFETCH, valid_percent=VALID_PERCENT)
    im_train, scores_train = train_ds # unzip
    im_valid, scores_valid = valid_ds # unzip
    
    if not ZERO_CENTER:
        im_train = (im_train+1.0)/2.0 # renormalize to [0, 1]
        im_valid = (im_valid+1.0)/2.0 # renormalize to [0, 1]
    
    # define placeholders
    im_pl = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, C, H, W]) # placeholder for images
    scores_pl = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, 1]) # placeholder for scores

    training_pl = tf.placeholder(dtype=tf.bool, shape=[])
    
    #model
    print("Building model ...")
    sys.stdout.flush()
    model = Scorer()
    scores_pred = model(inp=im_pl, training=training_pl, zero_centered=ZERO_CENTER)
    
#    print(scores_pred.shape)
#    sys.exit(0)
    
    # losses
    print("Losses ...")
    sys.stdout.flush()
    loss = model.compute_loss(scores_pl, scores_pred)

#    sys.exit(0)
    # define trainer
    print("Train_op ...")
    sys.stdout.flush()
    train_op, global_step = model.train_op(loss, LR, beta1=BETA1, beta2=BETA2)
    
#    sys.exit(0)
    
    # define summaries
    print("Summaries ...")
    sys.stdout.flush()
    train_loss_summary = tf.summary.scalar("train_loss", loss)
    valid_loss_pl = tf.placeholder(dtype=tf.float32, shape=[]) # placeholder for mean validation loss
    valid_loss_summary = tf.summary.scalar("valid_loss", valid_loss_pl)

    # summaries and graph writer
    print("Initializing summaries writer ...")
    sys.stdout.flush()
    if CONTINUE_TRAINING: # if continuing training, no need to write the graph again to the events file
        writer = tf.summary.FileWriter(CHECKPOINTS_PATH)
    else:
        writer = tf.summary.FileWriter(CHECKPOINTS_PATH, sess.graph)
    
    print("Initializing saver ...")
    sys.stdout.flush()
    saver = tf.train.Saver(tf.global_variables())
    if CONTINUE_TRAINING: # restore variables from saved model
        print("\nRestoring Model ...")
        saver.restore(sess, tf.train.latest_checkpoint(CHECKPOINTS_PATH)) # restore model from last checkpoint
        global_step_val = sess.run(global_step)
        for filename in glob.glob(os.path.join(CHECKPOINTS_PATH, "model*")): # remove all previously saved checkpoints (for limited disk space)
            os.remove(filename)
        saver.save(sess,os.path.join(CHECKPOINTS_PATH,"model"),global_step=global_step_val) # save the restored model (i,e keep the last checkpoint in this new run)
        print("Model restored from ", CHECKPOINTS_PATH)
        print("Continuing training for {} epochs ... ".format(NUM_EPOCHS))
        print("Global_step: {}\n".format(global_step_val))
        sys.stdout.flush()
    else: # initialize using initializers
        print("\nInitializing Variables")
        sys.stdout.flush()
        tf.global_variables_initializer().run()
    
    
    print("Train start ...")
    sys.stdout.flush()
    NUM_SAMPLES = nb_train
    NUM_VALID_SAMPLES = nb_valid
    best_valid_mae = None
    best_global_step = 0
#    sys.exit(0)
    with trange(int(NUM_EPOCHS * (NUM_SAMPLES // BATCH_SIZE))) as t:
        for i in t: # for each step
        
            # display training status
            epoch_cur = i * BATCH_SIZE/ NUM_SAMPLES # nb of epochs completed (e,g 1.5 => one epoch and a half)
            iter_cur = (i * BATCH_SIZE ) % NUM_SAMPLES # nb of images processed in current epoch
            t.set_postfix(epoch=epoch_cur,iter_percent="%d %%"%(iter_cur/float(NUM_SAMPLES)*100) )
            
            
            if (i+1) % LOG_ITER_FREQ == 0:
                im_val, scores_val = sess.run([im_train, scores_train]) # read data values from disk
                feed_dict_train={training_pl:True, im_pl: im_val, scores_pl: scores_val} # feed dict for training
                
                _, global_step_val, summary = sess.run([train_op, global_step, train_loss_summary], feed_dict_train) # train model and get loss_summary as well
                writer.add_summary(summary, global_step_val)
                
            else:
                im_val, scores_val = sess.run([im_train, scores_train]) # read data values from disk
                feed_dict_train={training_pl:True, im_pl: im_val, scores_pl: scores_val} # feed dict for training
                
                sess.run([train_op], feed_dict_train) # train model only (no summaries)

            
            # save model
            if (i+1) % SAVE_ITER_FREQ == 0:
                global_step_val = sess.run(global_step) # get the global step value
                saver.save(sess, os.path.join(CHECKPOINTS_PATH,"model"), global_step=global_step_val)
                gc.collect() # free-up memory once model saved
                
            if (i+1) % VALID_ITER_FREQ == 0:
                losses = []
                for j in range(NUM_VALID_SAMPLES//BATCH_SIZE):
                    im_val, scores_val = sess.run([im_valid, scores_valid]) # read data values from disk
                    feed_dict_valid={training_pl:False, im_pl: im_val, scores_pl: scores_val} # feed dict for validation
                
                    loss_val = sess.run(loss, feed_dict_valid)
                    losses.append(loss_val)
                
                loss_avg = np.array(losses).mean()
                
                summary, global_step_val = sess.run([valid_loss_summary, global_step], {valid_loss_pl: loss_avg})
                
                writer.add_summary(summary, global_step_val)
                
#                if best_valid_mae is None or loss_avg < best_valid_mae:
#                    best_valid_mae = loss_avg
#                    best_global_step = global_step_val
#                    saver.save(sess, os.path.join(CHECKPOINTS_PATH,"model"), global_step=global_step_val)
#                    print("new best validation mae: {}, at step: {}".format(best_valid_mae, best_global_step))
    
    global_step_val = sess.run(global_step) # get the global step value
    
    print("Training Done. Saving model ...")
    saver.save(sess, os.path.join(CHECKPOINTS_PATH,"model"), global_step=global_step_val) # save model 1 last time at the end of training
    print("Done with global_step_val: {}".format(global_step_val))
    
#    print("Training done with global_step_val: {}".format(global_step_val))
    
    
    
    
    
    
    
    
    
    
    
