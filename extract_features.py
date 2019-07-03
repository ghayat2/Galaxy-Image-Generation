import utils
import pathlib, os
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-d', '--image_dir', type=str, default="./generated_images/")
parser.add_argument('-o', '--out_dir', type=str, default="./manual_features/")
parser.add_argument('-p', '--prefix', type=str, default="")
parser.add_argument('-I', '--struct_from_img_dir', default=False, help="If True, will assume well-structured img_dir."
                                                                   " cf. run_experiment.py")
parser.add_argument('-m', '--max', type=int, help="set a maximum number of images from which to extract feats")
parser.add_argument('-f', '--force', help="Force to recompute all manual features even if already computed")
args = parser.parse_args()

def save_feats(out_dir, features, prefix):
    np.savetxt(os.path.join(out_dir, "{}_feats.gz".format(prefix)), features.astype(np.float32))
    # np.savetxt(os.path.join(out_dir, "{}_feats_ids.gz".format(prefix)), ids.astype(np.int32))

if args.struct_from_img_dir:
    for size in [64, 1000]:
        all_models = [model for model in pathlib.Path(os.path.join(args.image_dir, str(size))).glob("*")
                      if os.path.isdir(model)]
        all_models_name = [str(name).split('/')[-1] for name in all_models]
        for name in all_models_name:
            print("Extracting features for {} of size {}".format(name, size))
            if not os.path.isdir(os.path.join(args.out_dir, str(size), name)):
                os.makedirs(os.path.join(args.out_dir, str(size), name))
<<<<<<< HEAD
            # if len(os.listdir(os.path.join(args.out_dir, str(size), name))) > 0:
            #     print("Feature directory for model {} with size {} is not empty. Skipping...".format(name, size))
            # else:
            features, _, _, _ = utils.extract_features(image_dir=os.path.join(args.image_dir, str(size), name))
            save_feats(os.path.join(args.out_dir, str(size), name), features, name)
            # utils.extract_and_save_features(os.path.join(args.image_dir, str(size), name), name,
            #                                 os.path.join(args.out_dir, str(size)), args.max)
=======
            if len(os.listdir(os.path.join(args.out_dir, str(size), name))) > 0 and not args.force:
                print("Feature directory for model {} with size {} is not empty. Skipping...".format(name, size))
            else:
                feats, means, vars, _ = utils.extract_features(os.path.join(args.image_dir, str(size), name), args.max)
                np.savetxt(os.path.join(args.out_dir, str(size), name, "features_{}.gz".format(name)), feats)
                np.savetxt(os.path.join(args.out_dir, str(size), name, "means_{}.gz".format(name)), means)
                np.savetxt(os.path.join(args.out_dir, str(size), name, "vars_{}.gz".format(name)), vars)
>>>>>>> 22aded66e342b1d6c6d07065090526db1ac908ac
else:
    utils.extract_features(args.image_dir, args.max)

print("Manual Features from {} have been extracted".format(args.image_dir))