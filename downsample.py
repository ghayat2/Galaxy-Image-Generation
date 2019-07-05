import utils
import pathlib, os, sys
import numpy as np
from tqdm import tqdm
from PIL import Image
from argparse import ArgumentParser
import tensorflow as tf

parser = ArgumentParser()

parser.add_argument('-d', '--images_dir', type=str, required=True, help="the directory containing images to downsample")
parser.add_argument('-o', '--out_dir', type=str, default="./downsampled")

args = parser.parse_args()

IMAGES_DIR = args.images_dir
IMAGE_DIR_NAME = (IMAGES_DIR.rstrip("/")).split("/")[-1]
OUT_DIR = os.path.join(args.out_dir, IMAGE_DIR_NAME)

print("\n")
print("Run infos:")
print("    IMAGES_DIR: {}".format(IMAGES_DIR))
print("    OUT_DIR: {}".format(OUT_DIR))
print("\n")
sys.stdout.flush()

#sys.exit(0)

resizer = utils.make_max_pooling_resizer(vmin=0) # vmin=0 since by default pixel values read are in the range [0, 255]

if not os.path.isdir(OUT_DIR):
    os.makedirs(OUT_DIR)

all_images = [str(item) for item in pathlib.Path(IMAGES_DIR).glob('*')]

id = 0
for image in tqdm(all_images):
    decoded_image = np.array(Image.open(image))
    resized_image = resizer.predict(np.expand_dims(np.expand_dims(decoded_image, axis=0), axis=-1))
    resized_image = np.squeeze(resized_image).astype("uint8")
    resized_image = Image.fromarray(resized_image)
    resized_image.save(os.path.join(OUT_DIR, "image_" + str(id) + ".png"))
    id += 1

print("{} images have been resized and saved in {}".format(id, args.out_dir))
