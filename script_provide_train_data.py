import Image
import ImageOps
import numpy as np
import cPickle
import random
import threading
from os import listdir, remove
from os.path import isfile, join
from preprocess_image import *


class get_dataset(threading.Thread):
    def __init__ (self, thread_name, images_per_dump, mirror=False, max_dumps=20):
        threading.Thread.__init__(self)
        self.thread_name = thread_name
        self.images_per_dump = images_per_dump
        self.mirror = mirror
        self.max_dumps = max_dumps

    def run(self):
        images_train = np.zeros((self.images_per_dump, 224, 224, 3))
        labels_train = np.zeros((self.images_per_dump, 1000))

        file_number = 0
        dump_id = 0
        error_count = 0

        with open(description_path, 'r') as f:
            lines = f.readlines()
            while True:
            #for __ in range(1):
                sample_lines = random.sample(lines, self.images_per_dump)
                for line in sample_lines:
                    try:
                        joint_file_name, label_id = line.split(' ')
                        image_file_name = joint_file_name.split('/')[1]

                        #print '#', file_number, ', processing ', image_file_name

                        img = Image.open(image_path + image_file_name)
                        preprocessed_image = preprocess_image(img, self.mirror)

                        images_train[file_number] = preprocessed_image
                        labels_train[file_number, int(label_id)] = 1

                        file_number += 1
                    except:
                        error_count += 1

                np.savez(dump_path + 'image_net_preprocessed_dump_' + str(self.thread_name) + '_' + str(
                    dump_id) + '.npz', images_train, labels_train)
                dump_id += 1
                images_train = np.zeros((self.images_per_dump, 224, 224, 3))
                labels_train = np.zeros((self.images_per_dump, 1000))
                file_number = 0
                error_count = 0

                dump_files = [f for f in listdir(dump_path) if isfile(join(dump_path, f))]
                removed_files = 0
                if len(dump_files) > self.max_dumps:
                    excess_files = random.sample(dump_files, len(dump_files) - self.max_dumps)
                    for excess_file in excess_files:
                        try:
                            remove(dump_path+excess_file)
                            removed_files += 1
                        except:
                            pass


                print 'Thread', str(self.thread_name), ' processed ', dump_id, 'dump, ', error_count, 'errors. Removed ', removed_files, 'file(s).'


description_path = '/media/tassadar/Data/image_net/train_shuffled.txt'
image_path = '/media/tassadar/4870218670217BB6/image_net/img_train/'
dump_path = '/media/tassadar/Data/image_net/preprocessed_train/'

threads = []
for num in range(0, 4):
    thread = get_dataset(thread_name='Thread-'+str(num),
                         images_per_dump=500,
                         mirror=True,
                         max_dumps=20)
    thread.start()
    threads.append(thread)
