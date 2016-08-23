import Image
import ImageOps
import numpy as np
import random
from preprocess_image import *

description_path = '/media/tassadar/Data/image_net/val_shuffled.txt'
image_path = '/media/tassadar/4870218670217BB6/image_net/img_val/'
dump_path = '/media/tassadar/Data/image_net/preprocessed_val/'


def preprocess_cv_set(images_per_dump=5000, mirror=False):
    images_cv = np.zeros((images_per_dump, 224, 224, 3))
    labels_cv = np.zeros((images_per_dump, 1000))

    file_number = 0
    dump_id = 0
    error_count = 0

    with open(description_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            try:
                image_file_name, label_id = line.split(' ')

                print 'Dump ', dump_id, 'file', file_number, ', processing ', image_file_name

                img = Image.open(image_path + image_file_name)
                preprocessed_image = preprocess_image(img, mirror)

                images_cv[file_number] = preprocessed_image
                labels_cv[file_number, int(label_id)] = 1

                if file_number > 0 and file_number % (images_per_dump - 1) == 0:
                    np.savez(dump_path + 'image_net_val_preprocessed_dump_' + str(dump_id) + '.npz', images_cv,
                                 labels_cv)
                    print 'Processed ', dump_id, 'dump, ', error_count, 'errors.'
                    dump_id += 1
                    images_cv = np.zeros((images_per_dump, 224, 224, 3))
                    labels_cv = np.zeros((images_per_dump, 1000))
                    file_number = -1
                    error_count = 0

                file_number += 1
            except:
                error_count += 1


preprocess_cv_set(images_per_dump=2000)
