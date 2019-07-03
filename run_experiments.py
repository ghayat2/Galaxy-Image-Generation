import utils
import pathlib, os, json, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from argparse import ArgumentParser
from skimage.measure import shannon_entropy

parser = ArgumentParser()
parser.add_argument('-d', '--images_dir', type=str, default="./generated_images/")
parser.add_argument('-o', '--out_dir', type=str, default="./experiments_results/")
parser.add_argument('-f', '--feats_dir', type=str, default="./manual_features/")
parser.add_argument('-hm', '--heatmaps', help = '', action="store_true")
parser.add_argument('-knn', '--k_nearest_neighbors', help = '', action="store_true")
parser.add_argument('-box', '--box_plot', help = '', action="store_true")
parser.add_argument('-blobs', '--blobs', help = '', action="store_true")

parser.add_argument('-feats', '--features', help = '', action="store_true")
parser.add_argument('-all', '--all_experiments', help = 'To enable all experiments at once', action="store_true")



parser.add_argument('-bf', '--box_features', nargs='*', type=int, help="indices for the manual features to box_plot", required=False)
parser.add_argument('-l', '--legend', type=str, default=None, help="path to the legend json for the manual features", required=False)


args = parser.parse_args()

IMAGES_DIR = args.images_dir
FEATS_DIR = args.feats_dir
OUT_DIR = args.out_dir

ALL = args.all_experiments

HEATMAPS = args.heatmaps or ALL
KNN = args.k_nearest_neighbors or ALL
BOX = args.box_plot or ALL
FEATS = args.features or ALL
BLOBS = args.blobs or ALL


BOX_FEATS = args.box_features
LEGEND = args.legend

print("\n")
print("Run infos:")
print("    ALL_EXPERIMENTS: {}".format(ALL))
print("    HEATMAPS: {}".format(HEATMAPS))
print("    KNN: {}".format(KNN))
print("    BOX: {}".format(BOX))
print("    FEATS: {}".format(FEATS))
print("    BLOBS: {}".format(BLOBS))
print("    BOX_FEATS: {}".format(BOX_FEATS))
print("    LEGEND: {}".format(LEGEND))

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

# load the given models and image paths

if not os.path.isdir(FEATS_DIR):
    print("Features dir is not found")
if not os.path.isdir(IMAGES_DIR):
    print("Images dir is not found")
if not os.path.isdir(OUT_DIR):
    os.makedirs(OUT_DIR)

all_models = [model for model in pathlib.Path(os.path.join(IMAGES_DIR, "1000")).glob("*")]
all_models_name = [str(name).split('/')[-1] for name in all_models]
dir_1000 = os.path.join(IMAGES_DIR, "1000")
dir_64 = os.path.join(IMAGES_DIR, "64")

# load the json legend file if it exist
legend = {}
if LEGEND is not None:
    with open(LEGEND) as json_file:
        legend = json.load(json_file)

# run the heatmaps and save them in their respective directory
sns.set()

if BLOBS:
    for size in [64, 1000]:
        if not os.path.isdir(os.path.join(OUT_DIR, str(size))):
            os.mkdir(os.path.join(OUT_DIR, str(size)))
        all_models = [model for model in pathlib.Path(os.path.join(IMAGES_DIR, str(size))).glob("*")
                      if os.path.isdir(model)]
        all_models_name = [str(name).split('/')[-1] for name in all_models]
        for model_name in all_models_name:
            if not os.path.isdir(os.path.join(OUT_DIR, str(size), model_name)):
                os.mkdir(os.path.join(OUT_DIR, str(size), model_name))
            image_set = [image for image in pathlib.Path(os.path.join(IMAGES_DIR, str(size), model_name)).glob("*")]
            d_centers, mean_n_blobs, std_n_blobs = utils.blobs_summary(image_set)
            print((d_centers, mean_n_blobs, std_n_blobs))
        print("Blobs statistics done for size {}".format(size))

if HEATMAPS:
    for size in [64, 1000]:
        if not os.path.isdir(os.path.join(OUT_DIR, str(size))):
            os.mkdir(os.path.join(OUT_DIR, str(size)))
        all_models = [model for model in pathlib.Path(os.path.join(IMAGES_DIR, str(size))).glob("*")
                      if os.path.isdir(model)]
        all_models_name = [str(name).split('/')[-1] for name in all_models]
        for model_name in all_models_name:
            if not os.path.isdir(os.path.join(OUT_DIR, str(size), model_name)):
                os.mkdir(os.path.join(OUT_DIR, str(size), model_name))
            image_set = [image for image in pathlib.Path(os.path.join(IMAGES_DIR, str(size), model_name)).glob("*")]
            s = utils.heatmap(image_set, decode=True, shape=(size, size))
            print("Entropy for heatmap {} for size {} is {}".format(model_name, size, shannon_entropy(s)))
            h = sns.heatmap(s, cmap="gray")
            plt.annotate('entropy = {}'.format(shannon_entropy(s)), (0, 0), (0, -30), xycoords='axes fraction',
                                               textcoords='offset points', va='top')
            plt.savefig(os.path.join(OUT_DIR, str(size), model_name, "heatmap_{}_{}.png".format(model_name, size)))
            plt.close()
        print("Heatmaps generated for size {}".format(size))

if KNN:
    with open(os.path.join(OUT_DIR, "knn_results.txt"), "w+") as out:
        for size in [64]:
            for k in [1, 3, 5]:
                out.write("Images of size {}x{}\n".format(size, size))
                if not os.path.isdir(os.path.join(OUT_DIR, str(size))):
                    os.mkdir(os.path.join(OUT_DIR, str(size)))
                all_models = [model for model in pathlib.Path(os.path.join(IMAGES_DIR, str(size))).glob("*")
                              if os.path.isdir(model)]
                all_models_name = [str(name).split('/')[-1] for name in all_models]
                for model_name in all_models_name:
                    if not os.path.isdir(os.path.join(OUT_DIR, str(size), model_name)):
                        os.mkdir(os.path.join(OUT_DIR, str(size), model_name))
                    image_set = [image for image in pathlib.Path(os.path.join(IMAGES_DIR, str(size), model_name)).glob("*")]
                    stats = utils.leave_one_out_knn_diversity(image_set, size, k)
                    print("KNN stats for model {} with k = {} : (mean, std, vmin, vmax) = ({}, {}, {}, {})"
                          .format(model_name, k, *stats))
                    out.write("KNN stats for model {} with k = {} : (mean, std, vmin, vmax) = ({}, {}, {}, {})\n"
                              .format(model_name, k, *stats))
                print("KNN stats computed for size {}".format(size))

# Experiences on features
# Plot and save the box_plots and distplots
if BOX:
    for size in [64, 1000]:
        all_models = [model for model in pathlib.Path(os.path.join(IMAGES_DIR, str(size))).glob("*")
                      if os.path.isdir(model)]
        all_models_name = [str(name).split('/')[-1] for name in all_models]

        for model_name in all_models_name:
            if not os.path.isdir(os.path.join(OUT_DIR, str(size), model_name)):
                os.mkdir(os.path.join(OUT_DIR, str(size), model_name))
            features = np.loadtxt(os.path.join(FEATS_DIR, str(size), model_name, "features_{}.gz".format(model_name)))
            for f in BOX_FEATS:
                f = int(f)
                feat = features[:, f:f+1]
                label = legend[str(f)] if str(f) in legend.keys() else f
                plt.boxplot(feat, labels=[label])
                plt.savefig(os.path.join(OUT_DIR, str(size), model_name,
                                         "boxplot_{}_{}_feat{}.png".format(model_name, size, f)))
                plt.close()
                try:
                    sns.distplot(feat)
                    plt.title("Distribution for feature {}".format(label))
                    plt.savefig(os.path.join(OUT_DIR, str(size), model_name,
                                             "distplot_{}_{}_feat{}.png".format(model_name, size, f)))
                    plt.close()
                except np.linalg.LinAlgError:
                    print("Dist plot for {} could not be computed: the feature vector is singular".format(model_name))


    print("Boxplots and Distplots generated")

# Summarize manual features statistics in a table
if FEATS:
    n_feats = 38
    columns = [legend[str(f)] if str(f) in legend.keys() else str(f) for f in range(n_feats)]
    index = []
    for size in [64, 1000]:
        all_models = [model for model in pathlib.Path(os.path.join(IMAGES_DIR, str(size))).glob("*")
                      if os.path.isdir(model)]
        all_models_name = [str(name).split('/')[-1] for name in all_models]
        for model_name in all_models_name:
            index.append("{}_{}".format(model_name, size))

    stats_summary_mean = pd.DataFrame(index=index, columns=columns)
    stats_summary_var = pd.DataFrame(index=index, columns=columns)

    for size in [64, 1000]:
        all_models = [model for model in pathlib.Path(os.path.join(IMAGES_DIR, str(size))).glob("*")
                      if os.path.isdir(model)]
        all_models_name = [str(name).split('/')[-1] for name in all_models]
        for model_name in all_models_name:
            if not os.path.isdir(os.path.join(OUT_DIR, str(size), model_name)):
                os.mkdir(os.path.join(OUT_DIR, str(size), model_name))
            features = np.loadtxt(os.path.join(FEATS_DIR, str(size), model_name, "{}_feats.gz".format(model_name)))
            for f in range(features.shape[1]):
                f = int(f)
                feat = features[:, f:f+1]
                f_label = legend[str(f)] if str(f) in legend.keys() else str(f)
                stats_summary_mean[f_label]["{}_{}".format(model_name, size)] = np.mean(feat)
                stats_summary_var[f_label]["{}_{}".format(model_name, size)] = np.var(feat)

    print(stats_summary_mean)
    print(stats_summary_var)

    stats_summary_mean.to_csv(os.path.join(OUT_DIR, "stats_summary_mean.csv"))
    stats_summary_var.to_csv(os.path.join(OUT_DIR, "stats_summary_var.csv"))
    print("Summary tables have been created")
