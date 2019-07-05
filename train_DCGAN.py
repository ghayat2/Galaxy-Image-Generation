import tensorflow as tf
import numpy as np
import sys, os, glob, gc
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from data import create_dataloader_train_labeled
from DCGAN import DCGAN
from tqdm import trange
from PIL import Image
import datetime, time
from argparse import ArgumentParser
import patoolib
from tools import *

global_seed=5

tf.random.set_random_seed(global_seed)
np.random.seed(global_seed)

parser = ArgumentParser()
parser.add_argument('-ne', '--num_epochs', type = int, default = 500, help = 'number of training epochs')
parser.add_argument('-bs', '--batch_size', type = int, default = 16, help = 'size of training batch')
parser.add_argument('-d_lr', '--disc_learning_rate', type = float, default = 2e-4, help = 'learning rate for the optimizer of discriminator')
parser.add_argument('-g_lr', '--gen_learning_rate', type = float, default = 2e-4, help = 'learning rate for the optimizer of generator')
parser.add_argument('-d_b1', '--disc_beta_1', type = float, default = 0.9, help = 'beta 1 for the optimizer of discriminator')
parser.add_argument('-d_b2', '--disc_beta_2', type = float, default = 0.999, help = 'beta 2 for the optimizer of discriminator')
parser.add_argument('-g_b1', '--gen_beta_1', type = float, default = 0.9, help = 'beta 1 for the optimizer of generator')
parser.add_argument('-g_b2', '--gen_beta_2', type = float, default = 0.999, help = 'beta 2 for the optimizer of generator')
parser.add_argument('-n_dim', '--noise_dim', type = int, default = 1000, help = 'the dimension of the noise input to the generator')
parser.add_argument('-mb', '--minibatch_discrimination', help = 'whether we want minibatch discrimination', action="store_true")
parser.add_argument('-ls', '--label_smoothing', help = 'whether we want label smoothing', action="store_true")
parser.add_argument('-rot', '--rotate', help = 'whether to augment data using rotations of 90 degrees', action="store_true")

parser.add_argument('-lf', '--log_iter_freq', type = int, default = 100, help = 'number of iterations between training logs')
parser.add_argument('-spf', '--sample_iter_freq', type = int, default = 100, help = 'number of iterations between sampling steps')
parser.add_argument('-svf', '--save_iter_freq', type = int, default = 1000, help = 'number of iterations between saving model checkpoints')

parser.add_argument('-bp', '--batches_to_prefetch', type = int, default = 2, help = 'number of batches to prefetch')
parser.add_argument('-ct', '--continue_training', help = 'whether to continue training from the last checkpoint of the last experiment or not', action="store_true")

args = parser.parse_args()

CURR_TIMESTAMP=timestamp()

NUM_EPOCHS=args.num_epochs
BATCH_SIZE=args.batch_size
BATCHES_TO_PREFETCH=args.batches_to_prefetch
G_LR = args.gen_learning_rate # learning rate for generator
D_LR = args.disc_learning_rate # learning rate for discriminator
G_BETA1 = args.gen_beta_1
G_BETA2 = args.gen_beta_2
D_BETA1 = args.disc_beta_1
D_BETA2 = args.disc_beta_2
MINIBATCH_DISCRIMINATION = args.minibatch_discrimination
LABEL_SMOOTHING = args.label_smoothing
LOG_ITER_FREQ = args.log_iter_freq # train loss logging frequency (in nb of steps)
SAVE_ITER_FREQ = args.save_iter_freq
SAMPLE_ITER_FREQ = args.sample_iter_freq
CONTINUE_TRAINING = args.continue_training
ROTATE = args.rotate

C, H, W = 1, 1000, 1000 # images dimensions
NOISE_DIM=args.noise_dim
FIG_SIZE = 20 # in inches

# paths
DATA_ROOT="./data"
CLUSTER_DATA_ROOT="/cluster/scratch/mamrani/data"
if os.path.exists(CLUSTER_DATA_ROOT):
    DATA_ROOT=CLUSTER_DATA_ROOT
LOG_DIR=os.path.join(".", "LOG_DCGAN", CURR_TIMESTAMP)
if CONTINUE_TRAINING: # continue training from last training experiment
    list_of_files = glob.glob(os.path.join(".", "LOG_DCGAN", "*"))
    LOG_DIR = max(list_of_files, key=os.path.getctime) # latest created dir for latest experiment will be our log path
CHECKPOINTS_PATH = os.path.join(LOG_DIR, "checkpoints")
SAMPLES_DIR = os.path.join(LOG_DIR, "test_samples")

sys.stdout = Logger(LOG_DIR)

l = device_lib.list_local_devices()
gpus_list = [(device.physical_device_desc, device.memory_limit) for device in l if device.device_type == "GPU"]
print("\nGPUs List:")
for info in gpus_list:
    print(info[0])
    print("memory limit:", info[1])

# printing parameters
print("\n")
print("Run infos:")
print("    NUM_EPOCHS: {}".format(NUM_EPOCHS))
print("    BATCH_SIZE: {}".format(BATCH_SIZE))
print("    LEARNING_RATE_D: {}".format(D_LR))
print("    LEARNING_RATE_G: {}".format(G_LR))
print("    BETA1_D: {}".format(D_BETA1))
print("    BETA2_D: {}".format(D_BETA2))
print("    BETA1_G: {}".format(G_BETA1))
print("    BETA2_G: {}".format(G_BETA2))
print("    USE_MINIBATCH: {}".format(MINIBATCH_DISCRIMINATION))
print("    USE_LABEL_SMOOTHING: {}".format(LABEL_SMOOTHING))
print("    NOISE_DIM: {}".format(NOISE_DIM))
print("    BATCHES_TO_PREFETCH: {}".format(BATCHES_TO_PREFETCH))
print("    LOG_ITER_FREQ: {}".format(LOG_ITER_FREQ))
print("    SAVE_ITER_FREQ: {}".format(SAVE_ITER_FREQ))
print("    SAMPLE_ITER_FREQ: {}".format(SAMPLE_ITER_FREQ))
print("    DATA_ROOT: {}".format(DATA_ROOT))
print("    LOG_DIR: {}".format(LOG_DIR))
print("    CONTINUE_TRAINING: {}".format(CONTINUE_TRAINING))
print("\n")
sys.stdout.flush()

files = ["data.py",
         "layers.py",
         "DCGAN.py",
         "train_DCGAN.py"
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
    real_im, _, nb_reals, _ = create_dataloader_train_labeled(data_root=DATA_ROOT, batch_size=BATCH_SIZE, batches_to_prefetch=BATCHES_TO_PREFETCH, all_data=False, rotate=ROTATE)
    
    # define noise and test data
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM], seed=global_seed, name="random_noise") # noise fed to generator
    
    noise_test = np.random.normal(0, 1, [BATCH_SIZE, NOISE_DIM]).astype("float32") # constant noise to see its evolution over time
    
    # define placeholders
    im_pl = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, C, H, W]) # placeholder for real images fed to discriminator
    noise_pl = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, NOISE_DIM]) # placeholder for noise fed to generator
    
    training_pl = tf.placeholder(dtype=tf.bool, shape=[])
    
    #model
    print("Building model ...")
    sys.stdout.flush()
    model= DCGAN()
    fake_im, _ = model.generator_model(noise=noise_pl, training=training_pl) # get fake images from generator
    fake_out_D, _ = model.discriminator_model(inp=fake_im, training=training_pl, minibatch=MINIBATCH_DISCRIMINATION) # get discriminator output on fake images
    real_out_D, _ = model.discriminator_model(inp=im_pl, training=training_pl, reuse=True, resize=True, minibatch=MINIBATCH_DISCRIMINATION) # get discriminator output on real images
    
#    sys.exit(0)
    # losses
    print("Losses ...")
    sys.stdout.flush()
    gen_loss = model.generator_loss(fake_out=fake_out_D, labels=tf.ones(shape=[BATCH_SIZE], dtype=tf.int32), label_smoothing=LABEL_SMOOTHING)
    discr_loss = model.discriminator_loss(fake_out=fake_out_D, real_out=real_out_D, label_smoothing=LABEL_SMOOTHING,
                                      fake_labels=tf.zeros(shape=[BATCH_SIZE], dtype=tf.int32), 
                                      real_labels=tf.ones(shape=[BATCH_SIZE], dtype=tf.int32))

    # define trainer
    print("Train_op ...")
    sys.stdout.flush()
    gen_vars = model.generator_vars()
    gen_train_op, gen_global_step = model.train_op(gen_loss, G_LR, beta1=G_BETA1, beta2=G_BETA2, var_list=gen_vars, scope="generator")
    discr_vars = model.discriminator_vars()
    discr_train_op, discr_global_step = model.train_op(discr_loss, D_LR, beta1=D_BETA1, beta2=D_BETA2, var_list=discr_vars, scope="discriminator")
    
#    sys.exit(0)
    
    # define summaries
    print("Summaries ...")
    sys.stdout.flush()
    gen_loss_summary = tf.summary.scalar("gen_loss", gen_loss)
    discr_loss_summary = tf.summary.scalar("discr_loss", discr_loss)
	
    fake_im_channels_last = (tf.transpose(fake_im, perm=[0, 2, 3, 1])+1)*128.0 # put in channels last and unnormalize
    test_summary = tf.summary.image("Test Image", fake_im_channels_last, max_outputs=BATCH_SIZE)
    
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
        global_step_val = sess.run(gen_global_step)
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
    
    
    print("Train start at {} ...".format(timestamp()))
    NUM_SAMPLES = nb_reals
    sys.stdout.flush()
    with trange(int(NUM_EPOCHS * (NUM_SAMPLES // BATCH_SIZE))) as t:
        for i in t: # for each step
        
            # display training status
            epoch_cur = i * BATCH_SIZE/ NUM_SAMPLES # nb of epochs completed (e,g 1.5 => one epoch and a half)
            iter_cur = (i * BATCH_SIZE ) % NUM_SAMPLES # nb of images processed in current epoch
            t.set_postfix(epoch=epoch_cur,iter_percent="%d %%"%(iter_cur/float(NUM_SAMPLES)*100) )
            
            
            
            if (i+1) % LOG_ITER_FREQ == 0:
                im_val, noise_val = sess.run([real_im, noise]) # read data values from disk
                feed_dict_train={training_pl:True, im_pl: im_val, noise_pl: noise_val} # feed dict for training
                
                _, summary = sess.run([discr_train_op, discr_loss_summary], feed_dict_train) # train D and get loss_summary as well
                global_step_val = sess.run(discr_global_step)
                writer.add_summary(summary, global_step_val)
                
                _, summary = sess.run([gen_train_op, gen_loss_summary], feed_dict_train) # train G and get loss_summary as well
                global_step_val = sess.run(gen_global_step)
                writer.add_summary(summary, global_step_val)
                
            else:
                im_val, noise_val = sess.run([real_im, noise]) # read data values from disk
                feed_dict_train={training_pl:True, im_pl: im_val, noise_pl: noise_val} # feed dict for training
                
                sess.run(discr_train_op, feed_dict_train) # train D only (no summaries)
                
                sess.run(gen_train_op, feed_dict_train) # train G only (no summaries)

            
            # save model
            if (i+1) % SAVE_ITER_FREQ == 0:
                global_step_val = sess.run(gen_global_step) # get the global step value
                saver.save(sess, os.path.join(CHECKPOINTS_PATH,"model"), global_step=global_step_val)
                gc.collect() # free-up memory once model saved
                
            if (i+1) % SAMPLE_ITER_FREQ == 0:
                feed_dict_test={training_pl:False, noise_pl: noise_test} # feed dict for testing
                 
                images, summary, global_step_val = sess.run([fake_im, test_summary, gen_global_step], feed_dict_test)
                 
                if not os.path.exists(SAMPLES_DIR):
                    os.makedirs(SAMPLES_DIR)
                    
                fig = plt.figure(figsize=(FIG_SIZE, FIG_SIZE)) # Create a new "fig_size" inches by "fig_size" inches figure as default figure
                lines = cols =int(np.ceil(np.sqrt(BATCH_SIZE)))
                
                for j, image in enumerate(images):
                    image = ((image+1)*128.0).transpose(1,2,0).astype("uint8")[:, :, 0] # unnormalize image and put channels_last and remove the channels dimension
                    plt.subplot(lines, cols, j+1) # consider the default figure as lines x cols grid and select the (i+1)th cell
                    min_val = image.min()
                    max_val = image.max()
                    plt.imshow(image, cmap='gray', vmin=0, vmax=255) # plot the image on the selected cell
                    plt.axis('off')
                    plt.title("min: {}, max: {}".format(min_val, max_val))
                fig.savefig(os.path.join(SAMPLES_DIR, "img_step_{}.png".format(global_step_val))) # save image to dir
                plt.close()
                
                writer.add_summary(summary, global_step_val)
                
    print("Training Done at {}. Saving model ...".format(timestamp()))
    global_step_val = sess.run(gen_global_step) # get the global step value
    saver.save(sess, os.path.join(CHECKPOINTS_PATH,"model"), global_step=global_step_val) # save model 1 last time at the end of training
    print("Done with global_step_val: {}".format(global_step_val))
    
    
    
    
    
    
    
    
    
    
    
