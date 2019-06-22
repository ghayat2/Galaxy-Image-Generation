import random
import numpy as np

def gan_preprocessing(image):
    rint = random.randint(1, 4)
    image = np.rot90(image, rint)
    image = image / 255.0
    image = (image - 0.5)/0.5
    return image


def vae_preprocessing(image):
    rint = random.randint(1, 4)
    image = np.rot90(image, rint)
    image = image / 255.0
    return image

