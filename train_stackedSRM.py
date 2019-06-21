import tensorflow as tf
import numpy as np
import sys, os, glob, gc
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from data import create_dataloader_train
from StackedSRM import StackedSRM
import layers
from tqdm import trange
from PIL import Image
import datetime, time
from argparse import ArgumentParser

global_seed=5

tf.random.set_random_seed(global_seed)
np.random.seed(global_seed)

parser = ArgumentParser()
parser.add_argument('-ne', '--num_epochs', type = int, default = 500, help = 'number of training epochs')
parser.add_argument('-bs', '--batch_size', type = int, default = 16, help = 'size of training batch')
parser.add_argument('-ns', '--nb_stacks', type = int, default = 1, help = 'number of stacks')
parser.add_argument('-lr', '--learning_rate', type = float, default = 1e-3, help = 'learning rate for the optimizer')

parser.add_argument('-lf', '--log_iter_freq', type = int, default = 100, help = 'number of iterations between training logs')
parser.add_argument('-spf', '--sample_iter_freq', type = int, default = 100, help = 'number of iterations between sampling steps')
parser.add_argument('-svf', '--save_iter_freq', type = int, default = 1000, help = 'number of iterations between saving model checkpoints')

parser.add_argument('-bp', '--batches_to_prefetch', type = int, default = 2, help = 'number of batches to prefetch')
parser.add_argument('-ct', '--continue_training', help = 'whether to continue training from the last checkpoint of the last experiment or not', action="store_true")
#parser.add_argument('-c', '--colab', help = 'whether we are running on colab or not', action="store_true") # add this option to specify that the code is run on colab

args = parser.parse_args()

def timestamp():
    return datetime.datetime.fromtimestamp(time.time()).strftime("%Y.%m.%d-%H:%M:%S")

CURR_TIMESTAMP=timestamp()

NUM_EPOCHS=args.num_epochs
BATCH_SIZE=args.batch_size
BATCHES_TO_PREFETCH=args.batches_to_prefetch
LR = args.learning_rate # learning rate
NB_STACKS=args.nb_stacks

LOG_ITER_FREQ = args.log_iter_freq # train loss logging frequency (in nb of steps)
SAVE_ITER_FREQ = args.save_iter_freq
SAMPLE_ITER_FREQ = args.sample_iter_freq
CONTINUE_TRAINING = args.continue_training

FIG_SIZE = 20 # in inches
#RUNNING_ON_COLAB = args.colab

# paths
DATA_ROOT="./data"
CLUSTER_DATA_ROOT="/cluster/scratch/mamrani/data"
if os.path.exists(CLUSTER_DATA_ROOT):
    DATA_ROOT=CLUSTER_DATA_ROOT
LOG_DIR=os.path.join(".", "LOG_SRM", CURR_TIMESTAMP)
if CONTINUE_TRAINING: # continue training from last training experiment
    list_of_files = glob.glob(os.path.join(".", "LOG_SRM", "*"))
    LOG_DIR = max(list_of_files, key=os.path.getctime) # latest created dir for latest experiment will be our log path
CHECKPOINTS_PATH = os.path.join(LOG_DIR, "checkpoints")
SAMPLES_DIR = os.path.join(LOG_DIR, "samples")

# printing parameters
print("\n")
print("Run infos:")
print("    NUM_EPOCHS: {}".format(NUM_EPOCHS))
print("    BATCH_SIZE: {}".format(BATCH_SIZE))
print("    NB_STACKS: {}".format(NB_STACKS))
print("    LEARNING_RATE: {}".format(LR))
print("    BATCHES_TO_PREFETCH: {}".format(BATCHES_TO_PREFETCH))
print("    LOG_ITER_FREQ: {}".format(LOG_ITER_FREQ))
print("    SAVE_ITER_FREQ: {}".format(SAVE_ITER_FREQ))
print("    SAMPLE_ITER_FREQ: {}".format(SAMPLE_ITER_FREQ))
print("    DATA_ROOT: {}".format(DATA_ROOT))
print("    LOG_DIR: {}".format(LOG_DIR))
print("    CONTINUE_TRAINING: {}".format(CONTINUE_TRAINING))
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

    # data
    real_im, _, nb_reals, _ = create_dataloader_train(data_root=DATA_ROOT, batch_size=BATCH_SIZE, batches_to_prefetch=BATCHES_TO_PREFETCH, all_data=False)

    training_pl = tf.placeholder(dtype=tf.bool, shape=[])
    
    real_im = (real_im+1)/2 # renormalize images to the range [0, 1]
    # image preprocessing
    padded    = layers.padding_layer(real_im, padding=(12, 12), pad_values=0) # 1024x1024
    max_pool1 = layers.max_pool_layer(padded, pool_size=(2,2), strides=(2,2), pad_values=0) # 512x512
    max_pool2 = layers.max_pool_layer(max_pool1, pool_size=(2,2), strides=(2,2), pad_values=0) # 256x256
    max_pool3 = layers.max_pool_layer(max_pool2, pool_size=(2,2), strides=(2,2), pad_values=0) # 128x128
    max_pool4 = layers.max_pool_layer(max_pool3, pool_size=(2,2), strides=(2,2), pad_values=0) # 64x64
    
    outputs_gt = [max_pool3, max_pool2, max_pool1, padded] # outputs to predict
    
    #model
    print("Building model ...")
    sys.stdout.flush()
    model= StackedSRM(NB_STACKS)
    outputs_pred = model(max_pool4, training_pl)

#    for output in outputs_pred:
#        print(output.shape)
#    sys.exit(0)
    # losses
    print("Losses ...")
    sys.stdout.flush()
    loss = model.compute_loss(outputs_gt, outputs_pred)
    
#    sys.exit(0)
    # define trainer
    print("Train_op ...")
    sys.stdout.flush()
    train_op, global_step = model.train_op(loss, LR, beta1=0.5, beta2=0.999)
    
#    sys.exit(0)
    
    # define summaries
    print("Summaries ...")
    sys.stdout.flush()
    train_loss_summary = tf.summary.scalar("train_loss", loss)
    
#    sys.exit(0)
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
    NUM_SAMPLES = nb_reals
    sys.stdout.flush()
#    sys.exit(0)
    with trange(int(NUM_EPOCHS * (NUM_SAMPLES // BATCH_SIZE))) as t:
        for i in t: # for each step
        
            # display training status
            epoch_cur = i * BATCH_SIZE/ NUM_SAMPLES # nb of epochs completed (e,g 1.5 => one epoch and a half)
            iter_cur = (i * BATCH_SIZE ) % NUM_SAMPLES # nb of images processed in current epoch
            t.set_postfix(epoch=epoch_cur,iter_percent="%d %%"%(iter_cur/float(NUM_SAMPLES)*100) )
            
            
            
            if (i+1) % LOG_ITER_FREQ == 0:
                _, global_step_val, summary = sess.run([train_op, global_step, train_loss_summary], {training_pl:True}) # perform a train_step and get loss summary
                writer.add_summary(summary, global_step_val)
                
            else:
                sess.run([train_op], {training_pl:True}) # train_step only (no summaries)

            
            # save model
            if (i+1) % SAVE_ITER_FREQ == 0:
                global_step_val = sess.run(global_step) # get the global step value
                saver.save(sess, os.path.join(CHECKPOINTS_PATH,"model"), global_step=global_step_val)
                gc.collect() # free-up memory once model saved
                
            if (i+1) % SAMPLE_ITER_FREQ == 0:
                input_imgs, outputs_imgs_gt, outputs_imgs_pred, global_step_val = sess.run([max_pool4, outputs_gt, outputs_pred, global_step], {training_pl:False})
                
                images_batches_pred = [input_imgs] + outputs_imgs_pred # concat lists
                images_batches_gt = [input_imgs] + outputs_imgs_gt
                
                index = 0 # index of the image to show
                
                if not os.path.exists(SAMPLES_DIR):
                    os.makedirs(SAMPLES_DIR)
                    
                fig = plt.figure(figsize=(FIG_SIZE, FIG_SIZE)) # Create a new "fig_size" inches by "fig_size" inches figure as default figure
                lines = len(images_batches_pred)
                cols = 2

                for j in range(len(images_batches_pred)):
                    if j == 0:
                        image_gt = ((images_batches_gt[j][index])*255.0).transpose(1,2,0).astype("uint8")[:, :, 0] # unnormalize image and put channels_last and remove the channels dimension
                        image_pred = image_gt
                    else:
                        image_gt = (images_batches_gt[j][index]*255.0).transpose(1,2,0).astype("uint8")[:, :, 0] # unnormalize image and put channels_last and remove the channels dimension
                        image_pred = (images_batches_pred[j][index]*255.0).transpose(1,2,0).astype("uint8")[:, :, 0] # unnormalize image and put channels_last and remove the channels dimension
                    
                    # plot gt image
                    plt.subplot(lines, cols, 2*j+1) # consider the default figure as lines x cols grid and select the (j+1)th cell
                    min_val = image_gt.min()
                    max_val = image_gt.max()
                    plt.imshow(image_gt, cmap='gray', vmin=0, vmax=255) # plot the image on the selected cell
                    plt.title("min: {}, max: {}".format(min_val, max_val))
                    
                    # plot predicted image
                    plt.subplot(lines, cols, 2*j+2) # consider the default figure as lines x cols grid and select the (j+1)th cell
                    min_val = image_pred.min()
                    max_val = image_pred.max()
                    plt.imshow(image_pred, cmap='gray', vmin=0, vmax=255) # plot the image on the selected cell
                    plt.title("min: {}, max: {}".format(min_val, max_val))
                fig.savefig(os.path.join(SAMPLES_DIR, "img_step_{}.png".format(global_step_val))) # save image to dir
                plt.close()
                
    print("Training Done. Saving model ...")
    global_step_val = sess.run(global_step) # get the global step value
    saver.save(sess, os.path.join(CHECKPOINTS_PATH,"model"), global_step=global_step_val) # save model 1 last time at the end of training
    print("Done with global_step_val: {}".format(global_step_val))
    
    
    
    
    
    
    
    
    
    
    
