import numpy as np
import Image
import ImageOps
import random

def preprocess_image(img, mirror=False):
    height, width = img.size
    if height < width:
        new_height = 256
        new_width = int(width * (1. * new_height / height))
    else:
        new_width = 256
        new_height = int(height * (1. * new_width / width))
    resized = img.resize((new_height, new_width))

    if mirror and random.random > 0.5:
        resized = ImageOps.mirror(resized)

    resized = np.asarray(resized)
    #print 'Before: ', resized.shape

    mean_matrix = np.ones(resized.shape)
    mean_matrix[:, :, 0] = mean_matrix[:, :, 0] * 104.
    mean_matrix[:, :, 1] = mean_matrix[:, :, 1] * 117.
    mean_matrix[:, :, 2] = mean_matrix[:, :, 2] * 123.

    resized = np.true_divide(1. * resized - mean_matrix, mean_matrix)

    #resized = np.true_divide(1. * resized - np.array([104., 117., 123.]), np.array([104., 117., 123.]))

    #resized = np.true_divide(1. * resized - 128., 128.)

    #resized[:, :, 0] = np.true_divide(1. * resized[:, :, 0] - 104., 104.)
    #resized[:, :, 1] = np.true_divide(1. * resized[:, :, 1] - 117., 117.)
    #resized[:, :, 2] = np.true_divide(1. * resized[:, :, 2] - 123., 123.)

    #print 'After: ', resized.shape

    height_shift = int((new_height - 224) / 2)
    width_shift = int((new_width - 224) / 2)
    new_img = resized[width_shift:width_shift + 224, height_shift:height_shift + 224, :]
    return new_img
