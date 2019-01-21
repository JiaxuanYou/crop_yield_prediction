import numpy as np
import tensorflow as tf
import threading
# from fetch_data_histogram import *
import sys
import matplotlib.pyplot as plt
import time
import scipy.misc
from datetime import datetime
# datetime.now().strftime('%Y-%m-%d %H:%M:%S')

class Config():
    B, W, H, C = 32, 32,32, 9
    train_step = 25000
    lr = 1e-3
    weight_decay = 0.005

    keep_prob = 0.25
    # load_path = '/atlas/u/jiaxuan/data/MODIS_data_county_processed_compressed/'
    load_path = "/atlas/u/jiaxuan/data/google_drive/img_output/"
    # load_path = "/atlas/u/jiaxuan/data/google_drive/img_full_output/"
    # save_path = '/atlas/u/jiaxuan/data/train_results/histogram_new/test21/'
    # save_path = '/atlas/u/jiaxuan/data/train_results/histogram_new/test22_optimize/'
    save_path = '/atlas/u/jiaxuan/data/train_results/final/corn_yearly/'


def conv2d(input_data, out_channels, filter_size,stride, in_channels=None, name="conv2d"):
    if not in_channels:
        in_channels = input_data.get_shape()[-1]
    with tf.variable_scope(name):
        W = tf.get_variable("W", [filter_size, filter_size, in_channels, out_channels],
                initializer=tf.contrib.layers.variance_scaling_initializer())
        b = tf.get_variable("b", [1, 1, 1, out_channels])
        return tf.nn.conv2d(input_data, W, [1, stride, stride, 1], "SAME") + b


def pool2d(input_data, ksize, name="pool2d"):
    with tf.variable_scope(name):
        return tf.nn.max_pool(input_data, [1, ksize, ksize, 1], [1, ksize, ksize, 1], "SAME")


def conv_relu_batch(input_data, out_channels, filter_size,stride, in_channels=None, name="crb"):
    with tf.variable_scope(name):
        a = conv2d(input_data, out_channels, filter_size, stride, in_channels)
        b = batch_normalization(a,axes=[0,1,2])
        r = tf.nn.relu(b)
        return r

def dense(input_data, H, N=None, name="dense"):
    if not N:
        N = input_data.get_shape()[-1]
    with tf.variable_scope(name):
        W = tf.get_variable("W", [N, H], initializer=tf.contrib.layers.variance_scaling_initializer())
        b = tf.get_variable("b", [1, H])
        return tf.matmul(input_data, W, name="matmul") + b

def batch_normalization(input_data, axes=[0], name="batch"):
    with tf.variable_scope(name):
        mean, variance = tf.nn.moments(input_data, axes, keep_dims=True, name="moments")
        return tf.nn.batch_normalization(input_data, mean, variance, None, None, 1e-6, name="batch")

class batch_norm(object):
    """Code modification of http://stackoverflow.com/a/33950177"""
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum

            self.ema = tf.train.ExponentialMovingAverage(decay=self.momentum)
            self.name = name

    def __call__(self, x, axes=[0,1,2], train=True):
        shape = x.get_shape().as_list()

        if train:
            with tf.variable_scope(self.name) as scope:
                self.beta = tf.get_variable("beta", [shape[-1]],
                                    initializer=tf.constant_initializer(0.))
                self.gamma = tf.get_variable("gamma", [shape[-1]],
                                    initializer=tf.random_normal_initializer(1., 0.02))

                batch_mean, batch_var = tf.nn.moments(x, axes, name='moments')
                ema_apply_op = self.ema.apply([batch_mean, batch_var])
                self.ema_mean, self.ema_var = self.ema.average(batch_mean), self.ema.average(batch_var)

                with tf.control_dependencies([ema_apply_op]):
                    mean, var = tf.identity(batch_mean), tf.identity(batch_var)
        else:
            mean, var = self.ema_mean, self.ema_var

        normed = tf.nn.batch_norm_with_global_normalization(
                x, mean, var, self.beta, self.gamma, self.epsilon, scale_after_normalization=True)

        return normed

class NeuralModel():
    def __init__(self, config, name):

        self.x = tf.placeholder(tf.float32, [None, config.W, config.H, config.C], name="x")
        self.y = tf.placeholder(tf.float32, [None])
        self.lr = tf.placeholder(tf.float32, [])
        self.keep_prob = tf.placeholder(tf.float32, [])
        # self.year = tf.placeholder(tf.float32, [None,1])
        # used for max image
        # self.image = tf.Variable(initial_value=init,name="image")

        self.conv1_1 = conv_relu_batch(self.x, 128, 3,1, name="conv1_1")
        conv1_1_d = tf.nn.dropout(self.conv1_1, self.keep_prob)
        conv1_2 = conv_relu_batch(conv1_1_d, 256, 3,2, name="conv1_2")
        conv1_2_d = tf.nn.dropout(conv1_2, self.keep_prob)

        conv2_1 = conv_relu_batch(conv1_2_d, 256, 3,1, name="conv2_1")
        conv2_1_d = tf.nn.dropout(conv2_1, self.keep_prob)
        conv2_2 = conv_relu_batch(conv2_1_d, 512, 3,2, name="conv2_2")
        conv2_2_d = tf.nn.dropout(conv2_2, self.keep_prob)

        conv3_1 = conv_relu_batch(conv2_2_d, 512, 3,1, name="conv3_1")
        conv3_1_d = tf.nn.dropout(conv3_1, self.keep_prob)
        conv3_2= conv_relu_batch(conv3_1_d, 1024, 3,2, name="conv3_2")
        conv3_2_d = tf.nn.dropout(conv3_2, self.keep_prob)


        dim = np.prod(conv3_2_d.get_shape().as_list()[1:])
        flattened = tf.reshape(conv3_2_d, [-1, dim])
        # flattened_d = tf.nn.dropout(flattened, 0.25)

        self.fc6 = dense(flattened, 1024, name="fc6")
        # self.fc6 = tf.concat(1, [self.fc6_img,self.year])


        self.logits = tf.squeeze(dense(self.fc6, 1, name="dense"))
        self.loss_err = tf.nn.l2_loss(self.logits - self.y)


        with tf.variable_scope('dense') as scope:
            scope.reuse_variables()
            self.dense_W = tf.get_variable('W')
            self.dense_B = tf.get_variable('b')
        with tf.variable_scope('conv1_1/conv2d') as scope:
            scope.reuse_variables()
            self.conv_W = tf.get_variable('W')
            self.conv_B = tf.get_variable('b')

        # L1 term
        self.loss_reg = tf.abs(tf.reduce_sum(self.logits - self.y))
        # soybean
        # alpha = 1.5
        # corn
        alpha = 5
        self.loss = self.loss_err+self.loss_reg*alpha

        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)



