import utils
import pathlib, os, sys, glob
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-d', '--images_dir', type=str, default="./generated_images/")
parser.add_argument('-o', '--out_dir', type=str, default="./manual_features/")

parser.add_argument('-m', '--max', type=int, help="set a maximum number of images from which to extract feats")
parser.add_argument('-f', '--force', help="Force to recompute all manual features even if already computed")
args = parser.parse_args()


IMAGES_DIR = args.images_dir
OUT_DIR = args.out_dir

MAX_IMGS = args.max
FORCE = args.force

print("\n")
print("Run infos:")
print("    MAX_IMGS: {}".format(MAX_IMGS))
print("    FORCE: {}".format(FORCE))
print("    IMAGES_DIR: {}".format(IMAGES_DIR))
print("    OUT_DIR: {}".format(OUT_DIR))

#sys.exit(0)

def save_feats(out_dir, features, prefix):
    np.savetxt(os.path.join(out_dir, "{}_feats.gz".format(prefix)), features.astype(np.float32))
    # np.savetxt(os.path.join(out_dir, "{}_feats_ids.gz".format(prefix)), ids.astype(np.int32))

all_models = [model for model in glob.glob(os.path.join(IMAGES_DIR, "*")) if os.path.isdir(model)]
all_models_name = [str(name).split('/')[-1] for name in all_models]
for size in [64, 1000]:
    for name in all_models_name:
        print("\nExtracting features for {} of size {}".format(name, size))
        model_in_dir = os.path.join(IMAGES_DIR, name, str(size))
        if not os.path.exists(model_in_dir):
                print("Images of size {} not found".format(size))
                continue
        model_out_dir = os.path.join(OUT_DIR, name, str(size))
        if not os.path.isdir(model_out_dir):
            os.makedirs(model_out_dir)
        features, means, vars, _ = utils.extract_features(image_dir=model_in_dir)
        print("Features shape: {}".format(features.shape))
        save_feats(model_out_dir, features, name)
        np.savetxt(os.path.join(model_out_dir, "means_{}.gz".format(name)), means)
        np.savetxt(os.path.join(model_out_dir, "vars_{}.gz".format(name)), vars)

print("Manual Features from {} have been extracted".format(IMAGES_DIR))




