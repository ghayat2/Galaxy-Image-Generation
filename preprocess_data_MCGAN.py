import os
import pandas as pd
import pathlib
from shutil import copyfile

def create_labeled_folders(data_path):
    """Creates two folders within the labeled folder of the cosmology_aux_data_170429 folder,
    separating images associated to a label of 0 and of 1
    :param str data_path: Containing the path of the cosmology_aux_data_170429 folder
    """
    name = "labeled"
    new_dir = "labeled_split"
    labeled_images_new_path = os.path.join(data_path, new_dir)
    if not os.path.exists(labeled_images_new_path):
        os.makedirs(labeled_images_new_path)
    else:
        print("split labeled images dir exists, nothing to do ...")
        return
    
    labels_path = os.path.join(data_path, name + ".csv")
    labels = pd.read_csv(labels_path, index_col=0, skiprows=1, header=None)
    id_to_label = labels.to_dict(orient="index")
    id_to_label = {k: v[1] for k, v in id_to_label.items()}

    labeled_images_path = os.path.join(data_path, name)
    labeled_images_path = pathlib.Path(labeled_images_path)
    onlyFiles = [f for f in os.listdir(labeled_images_path) if (os.path.isfile(os.path.join(labeled_images_path, f)) and (f != None))]
    all_indexes = [item.split('.')[0] for item in onlyFiles]
    all_indexes = filter(None, all_indexes)
    all_pairs = [[item, id_to_label[int(item)]] for item in all_indexes]

    
    # Add if does not exist
    if(~os.path.isdir(os.path.join(labeled_images_new_path, '0'))):
        os.mkdir(os.path.join(labeled_images_new_path, '0'))
    if(~os.path.isdir(os.path.join(labeled_images_new_path, '1'))):
        os.mkdir(os.path.join(labeled_images_new_path, '1'))

    
    for file, label in all_pairs:
        copyfile(os.path.join(labeled_images_path, file + '.png'), os.path.join(labeled_images_new_path, "{}".format(int(label)), file + '.png'))
