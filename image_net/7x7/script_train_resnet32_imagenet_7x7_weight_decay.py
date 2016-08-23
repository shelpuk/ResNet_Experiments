import models
import cPickle
import numpy as np
import tensorflow as tf
from os import listdir, remove
from os.path import isfile, join
import random
import time

def get_examples(dump_path):
    train_images = []
    train_labels = []

    while train_images == [] or train_labels == []:
        train_files = [f for f in listdir(dump_path) if isfile(join(dump_path, f))]
        train_file = random.choice(train_files)
        print 'Loading ', train_file
        try:
            payload = np.load(dump_path + train_file)
            train_images = payload['arr_0']
            train_labels = payload['arr_1']
        except:
            print 'Fail. Waiting 10 sec...'
            time.sleep(10)
    return train_images, train_labels


def get_loss_accuracy(examples, labels, num_examples_to_test, batch_size, sess):
    data_length = examples.shape[0]
    example_ids = list(np.random.random_integers(0, data_length - 1, num_examples_to_test))
    images_to_test = examples[example_ids]
    labels_to_test = labels[example_ids]

    sum_accuracy = 0
    sum_loss = 0
    for i in range(0, num_examples_to_test, batch_size):
        batch_accuracy, batch_loss = sess.run([accuracy, loss], feed_dict={
            tf_images: images_to_test[i:i + batch_size],
            tf_labels: labels_to_test[i:i + batch_size]
        })
        sum_accuracy += batch_accuracy
        sum_loss += batch_loss
    accuracy_final = sum_accuracy / (num_examples_to_test // batch_size)
    loss_final = sum_loss / (num_examples_to_test // batch_size)
    return loss_final, accuracy_final



graph = tf.Graph()
image_size_rows = 224
image_size_cols = 224
num_channels = 3
num_labels = 1000

batch_size = 50

with graph.as_default():
    tf_images = tf.placeholder(tf.float32, shape=(batch_size, image_size_rows, image_size_cols, num_channels))
    tf_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))

    predictions, logits, l2_loss_total = models.resnet(tf_images)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_labels))

    global_step = tf.Variable(0, trainable=False)

    starter_learning_rate = 0.1
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 150000, 0.1, staircase=True)

    weight_decay = 0.0001
    l2_loss = l2_loss_total*weight_decay

    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss + l2_loss, global_step=global_step)

    correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(tf_labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    saver = tf.train.Saver()

    #accuracy_summary = tf.scalar_summary("accuracy", accuracy)

train_dump = '/media/tassadar/Data/image_net/preprocessed_train/'
cv_dump = '/media/tassadar/Data/image_net/preprocessed_val/'
autosave_file = "autosave-resnet32-imagenet-3x3-weight-decay.ckpt"
statistics_file = "autosave-resnet32-imagenet-3x3-weight-decay.csv"

num_global_steps = 60001
global_step = 0
offset = 0
autosave_period = 100
cv_period_check = 200


with tf.Session(graph=graph) as session:
    f_full = open('full-'+statistics_file, 'w')
    f_summary = open('summary-' + statistics_file, 'w')
    f_full.write('iteration, loss\n')
    f_summary.write('iteration, train_loss, cv_loss, train_accuracy, cv_accuracy\n')

    tf.initialize_all_variables().run()

    #saver.restore(session, autosave_file)

    print('Initialized')

    while global_step < num_global_steps:
        train_images, train_labels = get_examples(train_dump)
        data_length = train_images.shape[0]


        for step_ in range(2 * data_length // batch_size):
            example_ids = list(np.random.random_integers(0, data_length - 1, batch_size))
            minibatch_images = train_images[example_ids]
            minibatch_labels = train_labels[example_ids]

            feed_dict = {tf_images : minibatch_images,
                         tf_labels : minibatch_labels}
            _, l = session.run([optimizer, loss], feed_dict=feed_dict)
            print('Step %d: train loss: %f' % (global_step, l))
            f_full.write(str(global_step)+','+str(l)+'\n')

            global_step += 1

            if global_step % autosave_period == 0: saver.save(session, autosave_file)
            if global_step % cv_period_check == 0:
                train_images, train_labels = get_examples(train_dump)
                train_loss, train_accuracy = get_loss_accuracy(examples=train_images,
                                                               labels=train_labels,
                                                               num_examples_to_test=data_length,
                                                               batch_size=batch_size,
                                                               sess=session)

                cv_images, cv_labels = get_examples(cv_dump)
                cv_data_length = cv_images.shape[0]
                cv_loss, cv_accuracy = get_loss_accuracy(examples=cv_images,
                                                               labels=cv_labels,
                                                               num_examples_to_test=cv_data_length,
                                                               batch_size=batch_size,
                                                               sess=session)


                print('-------- cross-validation ----------')
                print('Step %d: train loss: %f, cv loss: %f, train accuracy: %f, cv accuracy: %f' % (global_step, train_loss, cv_loss, train_accuracy, cv_accuracy))
                print('-'*45)
                f_summary.write(str(global_step) + ',' + str(train_loss) + ',' + str(cv_loss) + ',' + str(train_accuracy) + ',' + str(cv_accuracy) + '\n')

    f_full.close()
    f_summary.close()


