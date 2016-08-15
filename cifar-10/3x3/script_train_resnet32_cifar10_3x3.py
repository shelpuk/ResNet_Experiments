import models
import cPickle
import numpy as np
import tensorflow as tf

graph = tf.Graph()
image_size_rows = 32
image_size_cols = 32
num_channels = 3
num_labels = 10

batch_size = 100

with graph.as_default():
    tf_images = tf.placeholder(tf.float32, shape=(batch_size, image_size_rows, image_size_cols, num_channels))
    tf_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))

    predictions, logits, l2_loss_total = models.resnet(tf_images, 32)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_labels))

    global_step = tf.Variable(0, trainable=False)

    starter_learning_rate = 0.01
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 10000, 0.1, staircase=True)

    #weight_decay = 0.0001
    #l2_loss = l2_loss_total*weight_decay

    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step=global_step)

    correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(tf_labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    saver = tf.train.Saver()

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

    #accuracy_summary = tf.scalar_summary("accuracy", accuracy)

autosave_period = 100
cv_period_check = 50

num_steps = 60001
offset = 0

print 'Loading data...'

[train_images, train_labels] = cPickle.load(open("/media/tassadar/Data/cifar-10/cifar10_train_square", 'rb'))
[test_images, test_labels] = cPickle.load(open("/media/tassadar/Data/cifar-10/cifar10_test_square", 'rb'))

data_length = train_images.shape[0]

autosave_file = "autosave-resnet32-cifar10-3x3-weight-decay.ckpt"
statistics_file = "autosave-resnet32-cifar10-3x3-weight-decay.csv"

with tf.Session(graph=graph) as session:
  f_full = open('full-'+statistics_file, 'w')
  f_summary = open('summary-' + statistics_file, 'w')
  f_full.write('iteration, loss\n')
  f_summary.write('iteration, train_loss, cv_loss, train_accuracy, cv_accuracy\n')

  tf.initialize_all_variables().run()

  #saver.restore(session, autosave_file)

  print('Initialized')
  for step in range(offset, offset + num_steps):
    example_ids = list(np.random.random_integers(0, data_length - 1, batch_size))
    minibatch_images = train_images[example_ids]
    minibatch_labels = train_labels[example_ids]

    feed_dict = {tf_images : minibatch_images,
                 tf_labels : minibatch_labels}
    _, l = session.run([optimizer, loss], feed_dict=feed_dict)
    f_full.write(str(step)+','+str(l)+'\n')

    if step % autosave_period == 0: saver.save(session, autosave_file)
    if step % cv_period_check == 0:
        train_loss, train_accuracy = get_loss_accuracy(examples=train_images,
                                                       labels=train_labels,
                                                       num_examples_to_test=2000,
                                                       batch_size=batch_size,
                                                       sess=session)

        cv_loss, cv_accuracy = get_loss_accuracy(examples=test_images,
                                                       labels=test_labels,
                                                       num_examples_to_test=2000,
                                                       batch_size=batch_size,
                                                       sess=session)


        print('Step %d: train loss: %f, cv loss: %f, train accuracy: %f, cv accuracy: %f' % (step, train_loss, cv_loss, train_accuracy, cv_accuracy))
        f_summary.write(str(step) + ',' + str(train_loss) + ',' + str(cv_loss) + ',' + str(train_accuracy) + ',' + str(cv_accuracy) + '\n')

  f_full.close()
  f_summary.close()


