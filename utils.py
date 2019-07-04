import os, sys
import numpy as np
import tensorflow as tf
import pathlib
import pywt
import skimage
import sklearn as sk

from tqdm import tqdm

from skimage.feature import blob_doh, blob_log
from skimage.exposure import histogram
from skimage.feature import shape_index
from skimage.measure import shannon_entropy
from skimage import color, io

# ------------------------------------------------------------------ FEATURES EXTRACTION --------------------------------------------------------------------------
def get_hand_crafted(one_image):
    """ Extracts various features out of the given image
    :param array one_image: the image from which features are to be extracted
    :return: the features associated with this image
    :rtype: Numpy array of size (38, 1)
    """
    #Select wavelet decomposition level so as to have the
    #same number of approximation coefficients
    if(one_image.shape[0] == 1000):
        wavedec_level = 9
    elif(one_image.shape[0] == 64):
        wavedec_level = 5

    hist = histogram(one_image, nbins=20, normalize=True)
    features = hist[0]
    blob_lo = blob_log(one_image, max_sigma=2.5, min_sigma=1.5, num_sigma=5, threshold=0.05)
    shape_ind = shape_index(one_image)
    shape_hist = np.histogram(shape_ind, range=(-1, 1), bins=9)
    shan_ent = shannon_entropy(one_image)
    max_val = one_image.max()
    min_val = one_image.min()
    variance_val = np.var(one_image)
    wavelet_approx = pywt.wavedec2(one_image, 'haar', level=wavedec_level)[0].flatten()
    features = np.concatenate([features, [blob_lo.shape[0]], shape_hist[0], [shan_ent], [max_val], [min_val], [variance_val], wavelet_approx])
    return features

def blobs_summary(image_paths):
    """
    Compute some statistics on the blobs on the images
    :param image_paths: paths to the images
    :return: number of distinct centers, mean of the number of blobs per image, std of the number of blobs per image
    """
    centers = set()
    n_blobs = []
    for path in tqdm(image_paths):
        image = io.imread(path, as_gray=True)
        blobs = blob_log(image, max_sigma=2.5, min_sigma=1.5, num_sigma=5, threshold=0.05)
        n_blobs += [len(blobs)]
        for x, y, sigma in blobs:
            centers.add((x, y))
    return len(centers), np.mean(n_blobs), np.std(n_blobs)

def features_summary(image_set, decode=True, return_ids=True, resize=False):
    """ Extracts various features out of the given image
    :param array image_set: the list of paths to images used to extract the summary
    :param bool decode: whether the images need to be decoded or not
    :param bool return_ids: whether we wish to have the ids be part of the output
    :return: the summary associated with the images
    :rtype: Tuple of Numpy arrays
    """
    features = []
    ids = []
    resizer = make_max_pooling_resizer(vmin=0)
    for image in tqdm(image_set):
        if return_ids:
            ids.append(str(image).split("/")[-1].split(".")[0])
        if decode:
            image = color.rgb2gray(io.imread(image))
            image = image / 255.0
            if(resize):
                image = resizer.predict(np.expand_dims(np.expand_dims(image, axis=0), axis=-1))[0, :, :, 0]
        assert np.amax(image) <= 1 and np.amin(image) >= 0 # Image must be in the same range to be compared
        features.append(get_hand_crafted(image))
    features = np.array(features)

    # Compute mean and variance of the features
    mean_features = np.mean(np.copy(features), axis=0)
    var_features = np.var(np.copy(features), axis=0)

    return features, mean_features, var_features, np.array(ids)

def extract_features(image_dir, max_imgs=None, resize=False):
    """
    Extract manual features and summaries from images contained in dir and saves them in the out_directory
    :param str image_dir: the directory containing the images
    :param int max_imgs: defines the max number of images to consider, consider all if set to None
    """
    all_images = [str(item) for item in pathlib.Path(image_dir).glob('*')]
    if max_imgs and len(all_images) > max_imgs:
        all_images = all_images[:max_imgs]
    features, means, vars, ids = features_summary(all_images, resize=resize)
    return features, means, vars, ids


# ------------------------------------------------------------------ EXPERIMENTS --------------------------------------------------------------------------

def heatmap(images_set, decode=False, shape=(1000, 1000)):
    """
    Given an image set, summarized it into a mean image
    :param array images_set: the images to extract the heat map from
    :param decode: weather the image needs to be decoded and pre-processed
    :param shape: input shape
    :return: the mean image
    """
    sum = np.zeros(shape)
    for image in images_set:
        if decode:
            image = io.imread(image, as_gray=True)
        image = image / 255.0
        assert np.amax(image) <= 1 and np.amin(image) >= 0 # Image must be in the same range to be compared
        sum += image
    sum /= len(images_set)
    return sum

def knn_diversity_stats(training_set, generated_imgs, k=3):
    """
    Find the k=3 nearest neighnors of an image in the training set and
    returns the average distance
    :param array training_set: the training set of images according to which we will find the nearest neighbours
    :param array generated_imgs: the images whose nearest neighbours we wish to find
    """
    knn = sk.neighbors.NearestNeighbors(n_neighbors=k)
    knn.fit(training_set, y=np.zeros(shape=(len(training_set),)))

    dists, idxs = knn.kneighbors(generated_imgs)
    return np.average(dists)

def decode_images(images_paths, size):
    """
    Given a list of imgs paths, decoded them and return the array
    :param images_paths: list of paths
    :param size: size of the imgs
    :return: narray of shape [len(images_paths), size, size] encoded in [0,1] floats
    """
    images = np.empty(shape=(len(images_paths), size, size), dtype=np.float)
    for idx, i in enumerate(images_paths):
        decoded = io.imread(i, as_gray=True)
        decoded = decoded.astype(np.float)
        decoded /= 255.0
        assert np.amax(decoded) <= 1.0 and np.amin(decoded) >= 0.0 and decoded.dtype == np.float
        images[idx] = decoded
    return images

def leave_one_out_knn_diversity(images_paths, size, k=3):
    """
    summarize the distance to k closest images in the sets
    :note: put all images into memory so the number of images
    int the set should be reasonable
    :param images_paths: paths to the imgs
    :param size: size of the imgs
    :returns a tuple (mean, std, min, max) of the distances
    """
    loo = sk.model_selection.LeaveOneOut()
    images = decode_images(images_paths, size)
    images = np.reshape(images, [len(images), -1]) # Flatten for sklearn [#samples, #features] framework
    dists = []
    for train_idx, test_idx in tqdm(loo.split(images)):
        train = images[train_idx]
        test = images[test_idx]
        d = knn_diversity_stats(train, test, k)
        dists.append(d)
    return np.average(dists), np.std(dists), np.amin(dists), np.amax(dists)

def make_max_pooling_resizer(vmin=-1):
    """
    Keras resizer for resizing 1000x1000 images into 64x64 max_pooled images
    :return: the (Keras) resizer
    """
    resizer = tf.keras.Sequential(
        [tf.keras.layers.Lambda(lambda x: x - vmin),
         tf.keras.layers.ZeroPadding2D(padding=(12, 12)),
         tf.keras.layers.Lambda(lambda x: x + vmin),
         tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
         tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
         tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
         tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
         tf.keras.layers.Reshape(target_shape=(64, 64, 1))]
    )
    return resizer
    

