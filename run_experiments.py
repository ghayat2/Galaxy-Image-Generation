import utils
import pathlib, os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from argparse import ArgumentParser
from skimage.measure import shannon_entropy

parser = ArgumentParser()
parser.add_argument('-d', '--image_dir', type=str, default="./generated_images/")
parser.add_argument('-o', '--out_dir', type=str, default="./experiments_results/")
parser.add_argument('-f', '--feat_dir', type=str, default="./manual_features/")
parser.add_argument('--heatmaps', type=bool, default=True)
parser.add_argument('--box_features', nargs='*', type=int, help="indices for the manual features to box_plot", required=False)
parser.add_argument('--legend', type=str, help="path to the legend json for the manual features", required=False)


args = parser.parse_args()

# the generated images structure should be:
# generated_images/
#           |
#           ---> 64/                # images 64x64
#           |     |
#           |     ---> model_1/
#           |     |
#           |     ---> model_2/
#           |     |
#           |     ---> .../
#           |
#           ---> 1000/              # images 1000x1000
#                 |
#                 ---> model_1/
#                 |
#                 ---> model_2/
#                 |
#                ---> .../
#
# the features directory structure should be:
# manual_features/
# |
#           ---> 64/                # images 64x64
#           |     |
#           |     ---> model_1/
#           |     |
#           |     ---> model_2/
#           |     |
#           |     ---> .../
#           |
#           ---> 1000/              # images 1000x1000
#                 |
#                 ---> model_1/
#                 |
#                 ---> model_2/
#                 |
#                ---> .../

# load the given models and image paths

if not os.path.isdir(args.feat_dir):
    print("Feature dir is not found")
if not os.path.isdir(args.image_dir):
    print("Image dir is not found")
if not os.path.isdir(args.out_dir):
    os.mkdir(args.out_dir)

all_models = [model for model in pathlib.Path(os.path.join(args.image_dir, "1000")).glob("*")]
all_models_name = [str(name).split('/')[-1] for name in all_models]
dir_1000 = os.path.join(args.image_dir, "1000")
dir_64 = os.path.join(args.image_dir, "1000")

# load the json legend file if it exist
legend = {}
if args.legend is not None:
    with open(args.legend) as json_file:
        legend = json.load(json_file)

# run the heatmaps and save them in their respective directory
sns.set()

if args.heatmaps:
    for size in [64, 1000]:
        if not os.path.isdir(os.path.join(args.out_dir, str(size))):
            os.mkdir(os.path.join(args.out_dir, str(size)))
        all_models = [model for model in pathlib.Path(os.path.join(args.image_dir, str(size))).glob("*")
                      if os.path.isdir(model)]
        all_models_name = [str(name).split('/')[-1] for name in all_models]
        for model_name in all_models_name:
            if not os.path.isdir(os.path.join(args.out_dir, str(size), model_name)):
                os.mkdir(os.path.join(args.out_dir, str(size), model_name))
            image_set = [image for image in pathlib.Path(os.path.join(args.image_dir, str(size), model_name)).glob("*")]
            s = utils.heatmap(image_set, decode=True, shape=(size, size))
            print("Entropy for heatmap {} for size {} is {}".format(model_name, size, shannon_entropy(s)))
            h = sns.heatmap(s, cmap="gray")
            plt.annotate('entropy = {}'.format(shannon_entropy(s)), (0, 0), (0, -30), xycoords='axes fraction',
                                               textcoords='offset points', va='top')
            plt.savefig(os.path.join(args.out_dir, str(size), model_name, "heatmap_{}_{}.png".format(model_name, size)))
            plt.close()
        print("Heatmaps generated for size {}".format(size))

# Plot and save the box_plots and distplots
for size in [64, 1000]:
    all_models = [model for model in pathlib.Path(os.path.join(args.image_dir, str(size))).glob("*")
                  if os.path.isdir(model)]
    all_models_name = [str(name).split('/')[-1] for name in all_models]

    for model_name in all_models_name:
        if not os.path.isdir(os.path.join(args.out_dir, str(size), model_name)):
            os.mkdir(os.path.join(args.out_dir, str(size), model_name))
        features = np.loadtxt(os.path.join(args.feat_dir, str(size), model_name, "features_{}.gz".format(model_name)))
        for f in args.box_features:
            f = int(f)
            feat = features[:, f:f+1]
            label = legend[str(f)] if str(f) in legend.keys() else f
            plt.boxplot(feat, labels=[label])
            plt.savefig(os.path.join(args.out_dir, str(size), model_name,
                                     "boxplot_{}_{}_feat{}.png".format(model_name, size, f)))
            plt.close()
            try:
                sns.distplot(feat)
                plt.title("Distribution for feature {}".format(label))
                plt.savefig(os.path.join(args.out_dir, str(size), model_name,
                                         "distplot_{}_{}_feat{}.png".format(model_name, size, f)))
                plt.close()
            except np.linalg.LinAlgError:
                print("Dist plot for {} could not be computed: the feature vector is singular".format(model_name))


print("Boxplots and Distplots generated")

# Summarize manual features statistics in a table
n_feats = 34
columns = [legend[str(f)] if str(f) in legend.keys() else str(f) for f in range(n_feats)]
index = []
for size in [64, 1000]:
    all_models = [model for model in pathlib.Path(os.path.join(args.image_dir, str(size))).glob("*")
                  if os.path.isdir(model)]
    all_models_name = [str(name).split('/')[-1] for name in all_models]
    for model_name in all_models_name:
        index.append("{}_{}".format(model_name, size))

stats_summary_mean = pd.DataFrame(index=index, columns=columns)
stats_summary_var = pd.DataFrame(index=index, columns=columns)

for size in [64, 1000]:
    all_models = [model for model in pathlib.Path(os.path.join(args.image_dir, str(size))).glob("*")
                  if os.path.isdir(model)]
    all_models_name = [str(name).split('/')[-1] for name in all_models]
    for model_name in all_models_name:
        if not os.path.isdir(os.path.join(args.out_dir, str(size), model_name)):
            os.mkdir(os.path.join(args.out_dir, str(size), model_name))
        features = np.loadtxt(os.path.join(args.feat_dir, str(size), model_name, "features_{}.gz".format(model_name)))
        for f in range(features.shape[1]):
            f = int(f)
            feat = features[:, f:f+1]
            f_label = legend[str(f)] if str(f) in legend.keys() else str(f)
            stats_summary_mean[f_label]["{}_{}".format(model_name, size)] = np.mean(feat)
            stats_summary_var[f_label]["{}_{}".format(model_name, size)] = np.var(feat)

print(stats_summary_mean)
print(stats_summary_var)

stats_summary_mean.to_csv(os.path.join(args.out_dir, "stats_summary_mean.csv"))
stats_summary_var.to_csv(os.path.join(args.out_dir, "stats_summary_var.csv"))
print("Summary tables have been created")
