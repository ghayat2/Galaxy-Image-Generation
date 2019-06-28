import numpy as np
import sys, os, glob, gc
from tqdm import trange
from PIL import Image
import datetime, time
from argparse import ArgumentParser
import layers, data
import patoolib

global_seed=5
np.random.seed(global_seed)

parser = ArgumentParser()
parser.add_argument('-ps', '--patch_size', type = int, default=50, help = 'size of the image patch along both dimensions')
parser.add_argument('-to_gen', '--to_generate', type = int, default = 100, help = 'the number of samples to generate')

args = parser.parse_args()

def timestamp():
    return datetime.datetime.fromtimestamp(time.time()).strftime("%Y.%m.%d-%H:%M:%S")

CURR_TIMESTAMP=timestamp()

IMG_SIZE = 1000

PATCH_SIZE = args.patch_size

TO_GENERATE = args.to_generate

DATA_ROOT="./data"
CLUSTER_DATA_ROOT="/cluster/scratch/mamrani/data"
if os.path.exists(CLUSTER_DATA_ROOT):
    DATA_ROOT=CLUSTER_DATA_ROOT
    
LOG_DIR = os.path.join("./LOG_PATCHES", CURR_TIMESTAMP)
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
print("    PATCH_SIZE: {}".format(PATCH_SIZE))
print("    TO_GENERATE: {}".format(TO_GENERATE))
print("    LOG_DIR: {}".format(LOG_DIR))
print("\n")
sys.stdout.flush()

labels2paths = data.read_labels2paths(data_root=DATA_ROOT)

galaxies_paths = labels2paths[1.0] # paths to galaxies

NB_IMAGES = len(galaxies_paths)

if not os.path.exists(GENERATED_SAMPLES_DIR):
    os.makedirs(GENERATED_SAMPLES_DIR)

for i in range(TO_GENERATE):
    a = time.time()
    print("generating image {}".format(i))
    gen_image = np.empty([IMG_SIZE, IMG_SIZE])
    for j in range(0, IMG_SIZE, PATCH_SIZE):
        for k in range(0, IMG_SIZE, PATCH_SIZE):
            index = np.random.randint(NB_IMAGES)
#            print(index)
            im_val = np.array(Image.open(galaxies_paths[index]))
            gen_image[j:j+PATCH_SIZE, k:k+PATCH_SIZE] = im_val[j:j+PATCH_SIZE, k:k+PATCH_SIZE]
    
    gen_image = gen_image.astype("uint8")
    image = Image.fromarray(gen_image)
    filename = "img_{}".format(i)
    image.save(os.path.join(GENERATED_SAMPLES_DIR, filename+".png"))
    print("Done in {} s".format(time.time() - a))

print("Done.")



