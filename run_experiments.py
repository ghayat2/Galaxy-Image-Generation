import utils
import pathlib, os, json, sys, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from argparse import ArgumentParser
from skimage.measure import shannon_entropy

parser = ArgumentParser()
parser.add_argument('-d', '--images_dir', type=str, default="./generated_images/")
parser.add_argument('-o', '--out_dir', type=str, default="./experiments_results/")
parser.add_argument('-f', '--feats_dir', type=str, default="./manual_features/")
parser.add_argument('-hm', '--heatmaps', help = '', action="store_true")
parser.add_argument('-knn', '--k_nearest_neighbors', help = '', action="store_true")
parser.add_argument('-box', '--box_plot', help = '', action="store_true")
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


BOX_FEATS = args.box_features
LEGEND = args.legend

print("\n")
print("Run infos:")
print("    ALL_EXPERIMENTS: {}".format(ALL))
print("    HEATMAPS: {}".format(HEATMAPS))
print("    KNN: {}".format(KNN))
print("    BOX: {}".format(BOX))
print("    FEATS: {}".format(FEATS))
print("    BOX_FEATS: {}".format(BOX_FEATS))
print("    LEGEND: {}".format(LEGEND))
print("    IMAGES_DIR: {}".format(IMAGES_DIR))
print("    FEATS_DIR: {}".format(FEATS_DIR))
print("    OUT_DIR: {}".format(OUT_DIR))

#sys.exit(0)

# the generated images structure should be:
# generated_images/
#           |
#           ---> model_1/                
#           |     |
#           |     ---> 64/          # images 64x64
#           |     |
#           |     ---> 1000/        # images 1000x1000
#           |     
#           ---> model_2/                
#           |     |
#           |     ---> 64/          # images 64x64
#           |     |
#           |     ---> 1000/        # images 1000x1000
#           |
#           .....
#
# the features directory structure should be:
# manual_features/
#           |
#           ---> model_1/                
#           |     |
#           |     ---> 64/          # images 64x64
#           |     |
#           |     ---> 1000/        # images 1000x1000
#           |     
#           ---> model_2/                
#           |     |
#           |     ---> 64/          # images 64x64
#           |     |
#           |     ---> 1000/        # images 1000x1000
#           |
#           .....

# load the given models and image paths

if not os.path.isdir(FEATS_DIR):
    print("Features dir is not found")
if not os.path.isdir(IMAGES_DIR):
    print("Images dir is not found")
if not os.path.isdir(OUT_DIR):
    os.makedirs(OUT_DIR)

all_models = [model for model in glob.glob(os.path.join(IMAGES_DIR, "*")) if os.path.isdir(model)]
all_models_name = [str(name).split('/')[-1] for name in all_models]

#print(all_models)
#print(all_models_name)

#sys.exit(0)

# load the json legend file if it exist
legend = {}
if LEGEND is not None:
    with open(LEGEND) as json_file:
        legend = json.load(json_file)

# run the heatmaps and save them in their respective directory
sns.set()

if HEATMAPS:
    print("\n")
    for size in [64, 1000]:
        for model_name in all_models_name:
            model_in_dir = os.path.join(IMAGES_DIR, model_name, str(size))
            model_out_dir = os.path.join(OUT_DIR, model_name, str(size))
            if not os.path.isdir(model_out_dir):
                os.makedirs(model_out_dir)
            image_set = [image for image in glob.glob(os.path.join(model_in_dir, "*"))]
            s = utils.heatmap(image_set, decode=True, shape=(size, size))
            print("Entropy for heatmap {} for size {} is {}".format(model_name, size, shannon_entropy(s)))
            h = sns.heatmap(s, cmap="gray")
            plt.annotate('entropy = {}'.format(shannon_entropy(s)), (0, 0), (0, -30), xycoords='axes fraction',
                                               textcoords='offset points', va='top')
            plt.savefig(os.path.join(model_out_dir, "heatmap_{}_{}.png".format(model_name, size)))
            plt.close()
        print("Heatmaps generated for size {}".format(size))

if KNN:
    print("\n")
    with open(os.path.join(OUT_DIR, "knn_results.txt"), "w+") as out:
        for size in [64]:
            for k in [1, 3, 5]:
                out.write("Images of size {}x{}\n".format(size, size))
                for model_name in all_models_name:
                    model_in_dir = os.path.join(IMAGES_DIR, model_name, str(size))
                    model_out_dir = os.path.join(OUT_DIR, model_name, str(size))
                    if not os.path.isdir(model_out_dir):
                        os.makedirs(model_out_dir)
                    image_set = [image for image in glob.glob(os.path.join(model_in_dir, "*"))]
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
        for model_name in all_models_name:
            model_in_dir = os.path.join(FEATS_DIR, model_name, str(size))
            model_out_dir = os.path.join(OUT_DIR, model_name, str(size))
            if not os.path.isdir(model_out_dir):
                os.makedirs(model_out_dir)
            features = np.loadtxt(os.path.join(model_in_dir, "{}_feats.gz".format(model_name)))
            
            nb_feats = features.shape[1]
            if BOX_FEATS is None:
                features_indices = range(nb_feats)
                nb_used = nb_feats
            else:
                features_indices = set(BOX_FEATS) # remove redundance
                nb_used = len(BOX_FEATS)
                
            print("\nPlotting boxplot on {} features for images of size {}".format(nb_used, size))
            for f in tqdm(features_indices):
                f = int(f)
                if f >= nb_feats: # check that the index is correct
                    print("Feature {} unavailable".format(f))
                    continue
                feat = features[:, f:f+1]
                label = legend[str(f)] if str(f) in legend.keys() else f
                plt.boxplot(feat, labels=[label])
                plt.savefig(os.path.join(model_out_dir, "boxplot_{}_{}_feat_{}.png".format(model_name, size, f)))
                plt.close()
                try:
                    sns.distplot(feat)
                    plt.title("Distribution for feature {}".format(label))
                    plt.savefig(os.path.join(model_out_dir, "distplot_{}_{}_feat_{}.png".format(model_name, size, f)))
                    plt.close()
                except:
                    print("Dist plot for {} could not be computed: the feature vector is singular".format(model_name))

    print("Boxplots and Distplots generated")

# Summarize manual features statistics in a table
if FEATS:
    print("\n")
    n_feats = 38
    columns = [legend[str(f)] if str(f) in legend.keys() else str(f) for f in range(n_feats)]
    index = []
    for size in [64, 1000]:
        for model_name in all_models_name:
            index.append("{}_{}".format(model_name, size))

    stats_summary_mean = pd.DataFrame(index=index, columns=columns)
    stats_summary_var = pd.DataFrame(index=index, columns=columns)

    for size in [64, 1000]:
        for model_name in all_models_name:
            model_in_dir = os.path.join(FEATS_DIR, model_name, str(size))
            model_out_dir = os.path.join(OUT_DIR, model_name, str(size))
            if not os.path.isdir(model_out_dir):
                os.makedirs(model_out_dir)
            features = np.loadtxt(os.path.join(model_in_dir, "{}_feats.gz".format(model_name)))
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
