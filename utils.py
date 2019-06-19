import datetime, time

def gan_preprocessing(image):
    image = image / 255.0
    image = (image - 0.5) / 0.5
    return image


def vae_preprocessing(image):
    image = image / 255.0
    return image
    
def timestamp():
    return datetime.datetime.fromtimestamp(time.time()).strftime("%Y.%m.%d-%H:%M:%S")
