'''
Module for building and running the model
'''
import os
import sys
import time
import utils
import logging
import numpy as np
import tensorflow as tf
from matplotlib import pyplot
from nolearn.lasagne import BatchIterator


class Paths(object):
    def __init__(self, params):
        self.model_name = self.get_model_name(params)
        self.var_scope = self.get_variables_scope(params)
        self.root_path = os.getcwd() + '/models/' + self.model_name + '/'
        self.model_path = self.get_model_path()
        self.training_history_path = self.get_train_history_path()
        self.learning_curves_path = self.get_learning_curves_path()
        os.makedirs(self.root_path, exist_ok=True)

    def get_model_name(self, params):
        model_name = 'k{}d{}p{}_k{}d{}p{}_k{}d{}p{}_fc{}p{}'.format(
            params.conv1_k, params.conv1_d, params.conv1_p,
            params.conv2_k, params.conv2_d, params.conv2_p,
            params.conv3_k, params.conv3_d, params.conv3_p,
            params.fc4_size, params.fc4_p
        )
        model_name += '_lrdec' if params.learning_rate_decay else'_no-lrdec'
        model_name += '_l2' if params.l2_reg_enabled else '_no-l2'
        return model_name

    def get_variables_scope(self, params):
        var_scope = 'k{}d{}_k{}d{}_k{}d{}_fc{}_fc0'.format(
            params.conv1_k, params.conv1_d,
            params.conv2_k, params.conv2_d,
            params.conv3_k, params.conv3_d,
            params.fc4_size
        )
        return var_scope

    def get_model_path(self):
        return self.root_path + 'model.ckpt'

    def get_train_history_path(self):
        return self.root_path + 'train_history'
    
    def get_learning_curves_path(self):
        return self.root_path + 'learning_curves.png'

class EarlyStopping(object):
    def __init__(self, saver, session, patience=100, minimize=True):
        self.minimize = minimize
        self.saver = saver
        self.session = session
        self.patience = patience
        self.best_monitored_value = np.inf if minimize else 0.
        self.best_monitored_epoch = 0
        self.restore_path = None

    def __call__(self, value, epoch):
        if (self.minimize and value < self.best_monitored_value) or (not self.minimize and value > self.best_monitored_value):
            self.best_monitored_value = value
            self.best_monitored_epoch = epoch
            self.restore_path = self.saver.save(self.session, os.getcwd() + '/early_stopping_checkpoint')
        elif self.best_monitored_epoch + self.patience < epoch:
            if self.restore_path != None:
                self.saver.restore(self.session, self.restore_path)
            else:
                print('ERROR: Failed to restore session')
            return True
        return False

def fully_connected(input, size):
    weights = tf.get_variable('weights', shape=[input.get_shape()[1], size], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable('biases', shape=[size], initializer=tf.constant_initializer(0.0))
    return tf.matmul(input, weights) + biases

def fully_connected_relu(input, size):
    return tf.nn.relu(fully_connected(input, size))

def conv_relu(input, kernel_size, depth):
    weights = tf.get_variable('weights', shape=[kernel_size, kernel_size, input.get_shape()[3], depth], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable('biases', shape=[depth], initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(conv + biases)

def pool(input, size):
    return tf.nn.max_pool(input, ksize=[1, size, size, 1], strides=[1, size, size, 1], padding='SAME')



def log_parameters(log, params, train_size, valid_size, test_size):
    if params.resume_training:
        print('________RESUME TRAINING___________')
    print('_____________DATA________________')
    print('  Training set: {} examples'.format(train_size))
    print('Validation set: {} examples'.format(valid_size))
    print('   Testing set: {} examples'.format(test_size))
    print('    Batch size: {}'.format(params.batch_size))
    print('_____________MODEL_______________')
    print('__________ARCHITECTURE___________')
    print(' %-*s %-*s %-*s %-*s' % (10, '', 10, 'Type', 8, 'Size', 15, 'Dropout (keep p)'))
    print(' %-*s %-*s %-*s %-*s' % (10, 'Layer 1', 10, '{}x{} Conv'.format(params.conv1_k, params.conv1_k), 8, str(params.conv1_d), 15, str(params.conv1_p)))
    print(' %-*s %-*s %-*s %-*s' % (10, 'Layer 2', 10, '{}x{} Conv'.format(params.conv2_k, params.conv2_k), 8, str(params.conv2_d), 15, str(params.conv2_p)))
    print(' %-*s %-*s %-*s %-*s' % (10, 'Layer 3', 10, '{}x{} Conv'.format(params.conv3_k, params.conv3_k), 8, str(params.conv3_d), 15, str(params.conv3_p)))
    print(' %-*s %-*s %-*s %-*s' % (10, 'Layer 4', 10, 'FC', 8, str(params.fc4_size), 15, str(params.fc4_p)))

    print('___________PARAMETERS____________')
    print('Learning rate decay: ' + ('Enabled' if params.learning_rate_decay else 'Disabled (rate = {})'.format(params.learning_rate)))
    print('  L2 Regularization: ' + ('Enabled (lambda = {})'.format(params.l2_lambda) if params.l2_reg_enabled else 'Disabled'))
    print('     Early stopping: ' + ('Enabled (patience = {})'.format(params.early_stopping_patience) if params.early_stopping_enabled else 'Disabled'))
    print('   Resume old model: ' + ('Enabled' if params.resume_training else 'Disabled'))

def plot_curve(axis, params, train_column, valid_column, linewidth=2, train_linestyle='b-', valid_linestyle='g-'):
    model_history = np.load(Paths(params).train_history_path + '.npz')
    train_values = model_history[train_column]
    valid_values = model_history[valid_column]
    epochs = train_values.shape(0)
    x_axis = np.arange(epochs)
    axis.plot(x_axis[train_values > 0], train_values[train_values > 0], train_linestyle, linewidth=linewidth, label='train')
    axis.plot(x_axis[valid_values > 0], valid_values[valid_values > 0], valid_linestyle, linewidth=linewidth, label='valid')
    return epochs

def plot_learning_curves(params):
    curves_figure = pyplot.figure(figsize=(10, 4))
    axis = curves_figure.add_subplot(1, 2, 1)
    epochs_plotted = plot_curve(axis, params, train_column='train_accuracy_history', valid_column='valid_accuracy_history')
    pyplot.grid()
    pyplot.legend()
    pyplot.xlabel('epoch')
    pyplot.ylabel('accuracy')
    pyplot.xlim(0, epochs_plotted)
    pyplot.ylim(50., 115.)
    axis = curves_figure.add_subplot(1, 2, 2)
    epochs_plotted = plot_curve(axis, params, train_column='train_loss_history', valid_column='valid_loss_history')
    pyplot.grid()
    pyplot.legend()
    pyplot.xlabel('epoch')
    pyplot.ylabel('loss')
    pyplot.xlim(0, epochs_plotted)
    pyplot.ylim(0.0001, 10.)
    pyplot.yscale('log')
    pyplot.show()


def model_pass(input, params, is_training):
    with tf.variable_scope('conv1'):
        conv1 = conv_relu(input, kernel_size=params.conv1_k, depth=params.conv1_d)
    with tf.variable_scope('pool1'):
        pool1 = pool(conv1, size=2)
        pool1 = tf.cond(is_training, lambda: tf.nn.dropout(pool1, keep_prob=params.conv1_p), lambda: pool1)
    with tf.variable_scope('conv2'):
        conv2 = conv_relu(input, kernel_size=params.conv2_k, depth=params.conv2_d)
    with tf.variable_scope('pool2'):
        pool2 = pool(conv2, size=2)
        pool2 = tf.cond(is_training, lambda: tf.nn.dropout(pool2, keep_prob=params.conv2_p), lambda: pool2)
    with tf.variable_scope('conv3'):
        conv3 = conv_relu(input, kernel_size=params.conv3_k, depth=params.conv3_d)
    with tf.variable_scope('pool3'):
        pool3 = pool(conv3, size=2)
        pool3 = tf.cond(is_training, lambda: tf.nn.dropout(pool3, keep_prob=params.conv3_p), lambda: pool3)

    pool1 = pool(pool1, size=4)
    shape = pool1.get_shape().as_list()
    pool1 = tf.reshape(pool1, [-1, shape[1] * shape[2] * shape[3]])

    pool2 = pool(pool2, size=2)
    shape = pool2.get_shape().as_list()
    pool2 = tf.reshape(pool2, [-1, shape[1] * shape[2] * shape[3]])

 #   pool3 = pool(pool3, size=2) #not in code
    shape = pool3.get_shape().as_list()
    pool3 = tf.reshape(pool3, [-1, shape[1] * shape[2] * shape[3]])

    flattened = tf.concat(1, [pool1, pool2, pool3])

    with tf.variable_scope('fc4'):
        fc4 = fully_connected_relu(flattened, size=params.fc4_size)
        fc4 = tf.cond(is_training, lambda: tf.nn.dropout(fc4, keep_prob=params.fc4_p), lambda: fc4)
    with tf.variable_scope('out'):
        logits = fully_connected(fc4, size=params.num_classes)
    return logits




def train_model(params, X_train, y_train, X_valid, y_valid, X_test, y_test):
    paths = Paths(params)
    log = logging.getLogger()
    fl = logging.FileHandler('train.log')
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    fl.setFormatter(formatter)
    screen = logging.StreamHandler(sys.stdout)
    log.setLevel(logging.DEBUG)
    log.addHandler(fl)
    log.addHandler(screen)

    start = time.time()
    model_variable_scope = paths.var_scope
    log_parameters(log, params, y_train.shape[0], y_valid.shape[0], y_test.shape[0])

    graph = tf.Graph()
    with graph.as_default():
        tf_x_batch = tf.placeholder(tf.float32, shape=(None, params.image_size[0], params.image_size[1], 1))
        tf_y_batch = tf.placeholder(tf.float32, shape=(None, params.num_classes))
        is_training = tf.placeholder(tf.bool)
        current_epoch = tf.Variable(0, trainable=False)

        if params.learning_rate_decay:
            learning_rate = tf.train.exponential_decay(params.learning_rate, current_epoch, decay_steps=params.max_epochs, decay_rate=0.01)
        else:
            learning_rate = params.learning_rate

        with tf.variable_scope(model_variable_scope):
            logits = model_pass(tf_x_batch, params, is_training)
            if params.l2_reg_enabled:
                with tf.variable_scope('fc4', reuse=True):
                    l2_loss = tf.nn.l2_loss(tf.get_variable('weights'))
            else:
                l2_loss = 0
        predictions = tf.nn.softmax(logits)
        softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, tf_y_batch)
        loss = tf.reduce_mean(softmax_cross_entropy) + params.l2_lambda * l2_loss

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    with tf.Session(graph=graph)as session:
        session.run(tf.global_variables_initializer())

        def get_accuracy_and_loss_in_batches(X, y):
            p = []
            sce = []
            batch_iterator = BatchIterator(batch_size=128)
            for x_batch, y_batch in batch_iterator(X, y):
                [p_batch, sce_batch] = session.run([predictions, softmax_cross_entropy], feed_dict={
                    tf_x_batch : x_batch,
                    tf_y_batch : y_batch,
                    is_training : False
                })
                p.extend(p_batch)
                sce.extend(sce_batch)
            p = np.array(p)
            sce = np.array(sce)
            accuracy = 100.0 * np.sum(np.argmax(p, 1) == np.argmax(y, 1)) / p.shape[0]
            loss = np.mean(sce)
            return (accuracy, loss)

        if params.resume_training:
            try:
                tf.train.Saver().restore(session, paths.model_path)
            except Exception as e:
                print("Failed restoring previous model: File does not exist.")
                pass
        saver = tf.train.Saver()
        early_stopping = EarlyStopping(tf.train.Saver(), session, patience=params.early_stopping_patience, minimize=True)
        train_loss_history = np.empty([0], dtype=np.float32)
        train_accuracy_history = np.empty([0], dtype=np.float32)
        valid_loss_history = np.empty([0], dtype=np.float32)
        valid_accuracy_history = np.empty([0], dtype=np.float32)

        if params.max_epochs > 0:
            print("____________TRAINING______________") # changed from log to print
        else:
            print('____________TESTING_______________')# ditto
        print('Timestamp:' + utils.get_time_hhmmss())
        #log.sync()

        for epoch in range(params.max_epochs):
            current_epoch = epoch
            batch_iterator = BatchIterator(batch_size=params.batch_size, shuffle=True)
            for x_batch, y_batch in batch_iterator(X_train, y_train):
                session.run([optimizer], feed_dict={
                    tf_x_batch : x_batch,
                    tf_y_batch : y_batch,
                    is_training : True
                })
            if epoch % params.log_epoch == 0:
                valid_accuracy, valid_loss = get_accuracy_and_loss_in_batches(X_valid, y_valid)
                train_accuracy, train_loss = get_accuracy_and_loss_in_batches(X_train, y_train)

                if epoch % params.print_epoch == 0:
                    print("____________EPOCH %4d/%d______________" % (epoch, params.max_epochs))
                    print("     Train loss: %.8f, accuracy: %.2f%%" % (train_loss, train_accuracy))
                    print("Validation loss: %.8f, accuracy: %.2f%%" % (valid_loss, valid_accuracy))
                    print("      Best loss: %.8f, accuracy: %.2f%%" % (early_stopping.best_monitored_value, early_stopping.best_monitored_epoch))
                    print("   Elapsed time: " + utils.get_time_hhmmss(start))
                    print("      Timestamp: " + utils.get_time_hhmmss())
                    #log.sync
            else:
                valid_loss = 0.
                valid_accuracy = 0.
                train_loss = 0.
                train_accuracy = 0.
            valid_loss_history = np.append(valid_loss_history, [valid_loss])
            valid_accuracy_history = np.append(valid_accuracy_history, [valid_accuracy])
            train_loss_history = np.append(train_loss_history, [train_loss])
            train_accuracy_history = np.append(train_accuracy_history, [train_accuracy])

            if params.early_stopping_enabled:
                if valid_loss == 0:
                    _, valid_loss = get_accuracy_and_loss_in_batches(X_valid, y_valid)
                if early_stopping(valid_loss, epoch):
                    print("Early stopping.\nBest monitored loss was {:.8f} at epoch {}".format(early_stopping.best_monitored_value, early_stopping.best_monitored_epoch))
                    break
        test_accuracy, test_loss = get_accuracy_and_loss_in_batches(X_test, y_test)
        valid_accuracy, valid_loss = get_accuracy_and_loss_in_batches(X_valid, y_valid)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(" Valid loss: %.8f, accuracy = %.2f%%" % (valid_loss, valid_accuracy))
        print("  Test loss: %.8f, accuracy = %.2f%%" % (test_loss, test_accuracy))
        print(" Total time: " + utils.get_time_hhmmss(start))
        print("  Timestamp: " + utils.get_time_hhmmss())

        saved_model_path = saver.save(session, paths.model_path)
        print("Model file: "+ saved_model_path)
        np.savez(paths.train_history_path, train_loss_history=train_loss_history, train_accuracy_history=train_accuracy_history, valid_loss_history=valid_loss_history, valid_accuracy_history=valid_accuracy_history)
        print("Training history file:"+ paths.train_history_path)
        plot_learning_curves(params)
