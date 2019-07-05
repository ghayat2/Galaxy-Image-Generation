import numpy as np
import sys, os, glob, gc
from tqdm import trange
from PIL import Image
from argparse import ArgumentParser
import data
from tools import *

global_seed=5
np.random.seed(global_seed)

parser = ArgumentParser()
parser.add_argument('-ps', '--patch_size', type = int, default=50, help = 'size of the image patch along both dimensions')
parser.add_argument('-to_gen', '--to_generate', type = int, default = 100, help = 'the number of samples to generate')

args = parser.parse_args()

def get_patch(image, center, filter_size):
    y, x = center
    size_y, size_x = filter_size
    y_up = y - size_y//2
    x_left = x - size_x//2
    
#    print("y, x = {}, {}".format(y, x))
#    print("y_up, x_left = {}, {}".format(y_up, x_left))
    image_size_y, image_size_x = image.shape
    
    patch = np.zeros(filter_size)
    for j in range(size_y):
        for k in range(size_x):
            pos_y = y_up+j
            pos_x = x_left+k
            if pos_y >=0 and pos_y < image_size_y and pos_x >=0 and pos_x < image_size_x:
                patch[j, k] = image[pos_y, pos_x]
                
    return patch 

CURR_TIMESTAMP=timestamp()

IMG_SIZE = 1000

PATCH_SIZE = args.patch_size

TO_GENERATE = args.to_generate

DATA_ROOT="./data"
    
LOG_DIR = os.path.join("./LOG_PATCHES", CURR_TIMESTAMP)
GENERATED_SAMPLES_DIR= os.path.join(LOG_DIR, "generated_samples", "1000")

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

with trange(TO_GENERATE) as t:
    for i in t:
        gen_image = np.empty([IMG_SIZE, IMG_SIZE])
        for j in range(0, IMG_SIZE, PATCH_SIZE):
            for k in range(0, IMG_SIZE, PATCH_SIZE):
                index = np.random.randint(NB_IMAGES)

                im_val = np.array(Image.open(galaxies_paths[index]))
                gen_image[j:j+PATCH_SIZE, k:k+PATCH_SIZE] = im_val[j:j+PATCH_SIZE, k:k+PATCH_SIZE]
        
        smoothing_filter = (1/9) * np.array([[1,1,1],
                                             [1,1,1],
                                             [1,1,1]])
        # apply smoothing to lines
        for j in range(PATCH_SIZE, IMG_SIZE, PATCH_SIZE):
            for k in range(IMG_SIZE):
                # apply on current line
                patch = get_patch(gen_image, center=(j, k), filter_size=(3,3))
                gen_image[j, k] = np.sum(smoothing_filter*patch) # apply filter
                
                # apply on previous line
                patch = get_patch(gen_image, center=(j-1, k), filter_size=(3,3))
                gen_image[j-1, k] = np.sum(smoothing_filter*patch) # apply filter
                
                # apply on next line
                patch = get_patch(gen_image, center=(j+1, k), filter_size=(3,3))
                gen_image[j+1, k] = np.sum(smoothing_filter*patch) # apply filter
        
        # apply smoothing to columns 
        for k in range(PATCH_SIZE, IMG_SIZE, PATCH_SIZE):
            for j in range(IMG_SIZE):
                # apply on current column
                patch = get_patch(gen_image, center=(j, k), filter_size=(3,3))
                gen_image[j, k] = np.sum(smoothing_filter*patch) # apply filter
                
                # apply on previous column
                patch = get_patch(gen_image, center=(j, k-1), filter_size=(3,3))
                gen_image[j, k-1] = np.sum(smoothing_filter*patch) # apply filter
                
                # apply on next column
                patch = get_patch(gen_image, center=(j, k+1), filter_size=(3,3))
                gen_image[j, k+1] = np.sum(smoothing_filter*patch) # apply filter

        gen_image = gen_image.astype("uint8")
        image = Image.fromarray(gen_image)
        filename = "img_{}".format(i)
        image.save(os.path.join(GENERATED_SAMPLES_DIR, filename+".png"))

print("Done.")



