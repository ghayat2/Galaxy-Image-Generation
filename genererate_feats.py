import utils
import os, sys, glob
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-no_l', '--no_labeled', help = 'to avoid generating manual features for the labeled set', action="store_true")
parser.add_argument('-no_s', '--no_scored', help = 'to avoid generating manual features for the scored set', action="store_true")
parser.add_argument('-no_q', '--no_query', help = 'to avoid generating manual features for the query set', action="store_true")

args = parser.parse_args()

GEN_LABELED = not args.no_labeled
GEN_SCORED = not args.no_scored
GEN_QUERY = not args.no_query

DATA_ROOT="./data"
FEATURES_DIR = os.path.join(DATA_ROOT, "features")

# printing parameters
print("\n")
print("Run infos:")
print("    Generating features for labeled set: {}".format(GEN_LABELED))
print("    Generating features for scored set: {}".format(GEN_SCORED))
print("    Generating features for query set: {}".format(GEN_QUERY))
print("    FEATURES_DIR: {}".format(FEATURES_DIR))


if not os.path.exists(FEATURES_DIR):
    os.makedirs(FEATURES_DIR)
    
LABELED_FEATS_PATH = os.path.join(FEATURES_DIR, 'labeled_feats.gz')
LABELED_FEATS_IDS_PATH = os.path.join(FEATURES_DIR, 'labeled_feats_ids.gz')

SCORED_FEATS_PATH = os.path.join(FEATURES_DIR, 'scored_feats.gz')
SCORED_FEATS_IDS_PATH = os.path.join(FEATURES_DIR, 'scored_feats_ids.gz')

QUERY_FEATS_PATH = os.path.join(FEATURES_DIR, 'query_feats.gz')
QUERY_FEATS_IDS_PATH = os.path.join(FEATURES_DIR, 'query_feats_ids.gz')

print("\n")
if GEN_LABELED and (not os.path.exists(LABELED_FEATS_PATH) or not os.path.exists(LABELED_FEATS_IDS_PATH)):
    utils.extract_and_save_features(image_dir=os.path.join(DATA_ROOT, 'labeled'), prefix='labeled', out_dir=FEATURES_DIR)
elif GEN_LABELED:
    print("Found labeled set's features and ids at {}. Nothing to do.".format(FEATURES_DIR))

print("\n")
if GEN_SCORED and (not os.path.exists(SCORED_FEATS_PATH) or not os.path.exists(SCORED_FEATS_IDS_PATH)):
    utils.extract_and_save_features(image_dir=os.path.join(DATA_ROOT, 'scored'), prefix='scored', out_dir=FEATURES_DIR)
elif GEN_SCORED:
    print("Found scored set's features and ids at {}. Nothing to do.".format(FEATURES_DIR))

print("\n")    
if GEN_QUERY and (not os.path.exists(QUERY_FEATS_PATH) or not os.path.exists(QUERY_FEATS_IDS_PATH)):
    utils.extract_and_save_features(image_dir=os.path.join(DATA_ROOT, 'query'), prefix='query', out_dir=FEATURES_DIR)
elif GEN_QUERY:
    print("Found query set's features and ids at {}. Nothing to do.".format(FEATURES_DIR))

