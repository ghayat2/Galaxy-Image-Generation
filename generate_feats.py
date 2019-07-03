import utils
import os, sys, glob
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-no_l', '--no_labeled', help = 'to avoid generating manual features for the labeled set', action="store_true")
parser.add_argument('-no_s', '--no_scored', help = 'to avoid generating manual features for the scored set', action="store_true")
parser.add_argument('-no_q', '--no_query', help = 'to avoid generating manual features for the query set', action="store_true")
parser.add_argument('-rs', '--resize', help = 'extract features on 64 x 64 max pooled images', action="store_true")

args = parser.parse_args()

GEN_LABELED = not args.no_labeled
GEN_SCORED = not args.no_scored
GEN_QUERY = not args.no_query
RESIZING = args.resize

DATA_ROOT="./data"
FEATURES_DIR = os.path.join(DATA_ROOT, "features")

# printing parameters
print("\n")
print("Run infos:")
print("    Generating features for labeled set: {}".format(GEN_LABELED))
print("    Generating features for scored set: {}".format(GEN_SCORED))
print("    Generating features for query set: {}".format(GEN_QUERY))
print("    Resizing: {}".format(RESIZING))
print("    FEATURES_DIR: {}".format(FEATURES_DIR))



if not os.path.exists(FEATURES_DIR):
    os.makedirs(FEATURES_DIR)
    
LABELED_DIR = os.path.join(DATA_ROOT, 'labeled')
SCORED_DIR = os.path.join(DATA_ROOT, 'scored')
QUERY_DIR = os.path.join(DATA_ROOT, 'query')
    
LABELED_FEATS_PATH = os.path.join(FEATURES_DIR, 'labeled_feats.gz')
LABELED_FEATS_IDS_PATH = os.path.join(FEATURES_DIR, 'labeled_feats_ids.gz')

SCORED_FEATS_PATH = os.path.join(FEATURES_DIR, 'scored_feats.gz')
SCORED_FEATS_IDS_PATH = os.path.join(FEATURES_DIR, 'scored_feats_ids.gz')

QUERY_FEATS_PATH = os.path.join(FEATURES_DIR, 'query_feats.gz')
QUERY_FEATS_IDS_PATH = os.path.join(FEATURES_DIR, 'query_feats_ids.gz')

def save_feats(out_dir, features, ids, prefix):
    np.savetxt(os.path.join(out_dir, "{}_feats.gz".format(prefix)), features)
    np.savetxt(os.path.join(out_dir, "{}_feats_ids.gz".format(prefix)), ids)

# Start

# Generate features for labeled images
print("\n")
if GEN_LABELED and (not os.path.exists(LABELED_FEATS_PATH) or not os.path.exists(LABELED_FEATS_IDS_PATH)):
    features, _, _, ids = utils.extract_features(image_dir=LABELED_DIR, resize=RESIZING)
    ids = ids.astype(int) # convert id to int
    l_prefix = 'labeled'
    if(RESIZING):
        l_prefix = 'labeled_64'
    save_feats(out_dir=FEATURES_DIR, features=features, ids=ids, prefix=l_prefix)
elif GEN_LABELED:
    print("Found labeled set's features and ids at {}. Nothing to do.".format(FEATURES_DIR))

# Generate features for scored images
print("\n")
if GEN_SCORED and (not os.path.exists(SCORED_FEATS_PATH) or not os.path.exists(SCORED_FEATS_IDS_PATH)):
    features, _, _, ids = utils.extract_features(image_dir=SCORED_DIR, resize=RESIZING)
    ids = ids.astype(int) # convert id to int
    # read the scores to be include them in the saved features file
    scores = np.genfromtxt(os.path.join(DATA_ROOT, "scored.csv"), delimiter=",", skip_header=1)
    scores_dict = dict(zip(scores[:,0], scores[:,1]))
    scores_ordered = np.array([scores_dict[id] for id in ids]).reshape([-1, 1]) # get the scores per id in the same order as in ids
    
    features = np.concatenate([features, scores_ordered], axis=1) # add last column for scores
    s_prefix = 'scored'
    if(RESIZING):
        s_prefix = 'scored_64'
    save_feats(out_dir=FEATURES_DIR, features=features, ids=ids, prefix=s_prefix)
elif GEN_SCORED:
    print("Found scored set's features and ids at {}. Nothing to do.".format(FEATURES_DIR))

# Generate features for query images
print("\n")    
if GEN_QUERY and (not os.path.exists(QUERY_FEATS_PATH) or not os.path.exists(QUERY_FEATS_IDS_PATH)):
    features, _, _, ids = utils.extract_features(image_dir=QUERY_DIR, resize=RESIZING)
    ids = ids.astype(int) # convert id to int
    q_prefix = 'query'
    if(RESIZING):
        q_prefix = 'query_64'
    save_feats(out_dir=FEATURES_DIR, features=features, ids=ids, prefix='query')
elif GEN_QUERY:
    print("Found query set's features and ids at {}. Nothing to do.".format(FEATURES_DIR))

