import random

description_path = '/media/tassadar/Data/image_net/val.txt'
shuffled_path = '/media/tassadar/Data/image_net/val_shuffled.txt'

with open(description_path, 'r') as f:
    lines = f.readlines()
    num_images = len(lines)
    shuffled_ids = random.sample(range(0, num_images), num_images)
    with open(shuffled_path, 'w') as d:
        for i in range(num_images):
            if i % 1000 == 0: print i
            d.write(lines[shuffled_ids[i]])

print 'Complete. Shuffled ', num_images, 'lines.'
