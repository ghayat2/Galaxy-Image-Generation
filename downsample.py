import utils
import pathlib, os
import numpy as np
from tqdm import tqdm
from skimage import color, io
from argparse import ArgumentParser
import tensorflow as tf

parser = ArgumentParser()
parser.add_argument('-d', '--image_dir', type=str, default="./generated_images/")
parser.add_argument('-o', '--out_dir', type=str, default="./downsampled")
parser.add_argument('-v', '--vmin', type=int, default="0 if imgs in [0,1] range, -1 if images in [-1,1]")


args = parser.parse_args()

resizer = utils.make_max_pooling_resizer(args.vmin)

if not os.path.isdir(args.out_dir):
    os.mkdir(args.out_dir)

all_images = [str(item) for item in pathlib.Path(args.image_dir).glob('*')]
id = 0
for image in tqdm(all_images):
    decoded_image = color.rgb2gray(io.imread(image))
    resized_image = resizer(np.expand_dims(np.expand_dims(decoded_image, axis=0), axis=-1))
    resized_image = np.squeeze(resized_image)
    io.imsave(os.path.join(args.out_dir, "image_" + str(id) + ".png"), resized_image)
    id += 1

print("{} images have been resized and saved in {}".format(id, args.out_dir))