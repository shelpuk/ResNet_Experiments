import Image
import ImageOps
import numpy as np
import cPickle
import random
import threading


class get_dataset(threading.Thread):
    def __init__ (self, thread_name, images_per_dump, mirror=False):
        threading.Thread.__init__(self)
        self.thread_name = thread_name
        self.images_per_dump = images_per_dump
        self.mirror = mirror

    def run(self):
        images_train = np.zeros((self.images_per_dump, 3, 224, 224))
        labels_train = np.zeros((self.images_per_dump, 1000))

        file_number = 0
        dump_id = 0
        error_count = 0

        with open(description_path, 'r') as f:
            lines = f.readlines()
            while True:
                sample_lines = random.sample(lines, self.images_per_dump)
                for line in sample_lines:
                    try:
                        joint_file_name, label_id = line.split(' ')
                        image_file_name = joint_file_name.split('/')[1]

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
                images_train = np.zeros((self.images_per_dump, 3, 224, 224))
                labels_train = np.zeros((self.images_per_dump, 1000))
                file_number = 0
                error_count = 0

                print 'Thread', str(self.thread_name), ' processed ', dump_id, 'dump, ', error_count, 'errors.'

description_path = '/media/tassadar/Data/image_net/train_shuffled.txt'
image_path = '/media/tassadar/4870218670217BB6/image_net/img_train/'
dump_path = '/media/tassadar/4870218670217BB6/image_net/img_preprocessed/'



threads = []
for num in range(0, 4):
    thread = get_dataset('Thread-'+str(num), 100)
    thread.start()
    threads.append(thread)
