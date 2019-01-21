import numpy as np
import tensorflow as tf
import threading
from fetch_data_county import *
import sys
import matplotlib.pyplot as plt
import time
from datetime import datetime
# datetime.now().strftime('%Y-%m-%d %H:%M:%S')

class Config():
    B, W, H, C = 32, 32,32, 9

    lstm_layers = 1
    # 64
    # hidden 256(default)
    lstm_H = 128

    # dense 256(default)
    dense = 256

    train_step = 10000
    lr = 1e-3
    drop_out = 0.75
    # weight_decay = 0.005
    load_path = "/atlas/u/jiaxuan/data/google_drive/img_output/"
    save_path = '/atlas/u/jiaxuan/data/train_results/final/lstm/'
    # save_path = '~/Downloads/'

def conv2d(input_data, out_channels, filter_size, in_channels=None, name="conv2d"):
    if not in_channels:
        in_channels = input_data.get_shape()[-1]
    with tf.variable_scope(name):
        W = tf.get_variable("W", [filter_size, filter_size, in_channels, out_channels],
                initializer=tf.contrib.layers.variance_scaling_initializer())
        b = tf.get_variable("b", [1, 1, 1, out_channels])
        return tf.nn.conv2d(input_data, W, [1, 1, 1, 1], "SAME") + b

def pool2d(input_data, ksize, name="pool2d"):
    with tf.variable_scope(name):
        return tf.nn.max_pool(input_data, [1, ksize, ksize, 1], [1, ksize, ksize, 1], "SAME")


def conv_relu_batch(input_data, out_channels, filter_size, in_channels=None, name="crb"):
    with tf.variable_scope(name):
        a = conv2d(input_data, out_channels, filter_size, in_channels)
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


def lstm_net(input_data,output_data,config,keep_prob = 1,name='lstm_net'):
    with tf.variable_scope(name):
        lstm_cell = tf.nn.rnn_cell.LSTMCell(config.lstm_H,state_is_tuple=True)
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.lstm_layers,state_is_tuple=True)
        state = cell.zero_state(config.B, tf.float32)
        outputs, final_state = tf.nn.dynamic_rnn(cell, input_data, 
                       initial_state=state, time_major=True)
        output_final = tf.squeeze(tf.slice(outputs, [config.H-1,0,0] , [1,-1,-1]))
        # print outputs.get_shape().as_list()
        fc1 = dense(output_final, config.dense, name="dense")

        logit = tf.squeeze(dense(fc1,1,name='logit'))
        loss = tf.nn.l2_loss(logit - output_data)

        return logit,loss,fc1

class NeuralModel():
    def __init__(self, config, name):
        self.x = tf.placeholder(tf.float32, [None, config.W, config.H, config.C], name="x")
        self.y = tf.placeholder(tf.float32, [None])
        self.lr = tf.placeholder(tf.float32, [])
        self.keep_prob = tf.placeholder(tf.float32, [])

        input_data = tf.transpose(self.x, [2,0,1,3])
        dim = input_data.get_shape().as_list()
        input_data = tf.reshape(input_data,[dim[0],-1,dim[2]*dim[3]])
        print('lstm input shape',input_data.get_shape())

        with tf.variable_scope('LSTM') as scope:
            self.pred,self.loss,self.feature = lstm_net(input_data, self.y, config, keep_prob=self.keep_prob)

        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        with tf.variable_scope('LSTM/lstm_net/logit') as scope:
            scope.reuse_variables()
            self.dense_W = tf.get_variable('W')
            self.dense_B = tf.get_variable('b')

# if __name__ == '__main__':
#     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
#     sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
#     config = Config()
#     model = NeuralModel(config, "model")

#     dummy_x = np.random.rand(config.B, config.W, config.H, config.C)
#     dummy_y = np.random.rand(config.B)

#     sess.run(tf.initialize_all_variables())
#     for i in range(1000):
#         # model.state = model.cell.zero_state(config.B, tf.float32)
#         if i % 100 == 0:
#             config.lr /= 2
#         _, loss, pred = sess.run([model.train_op, model.loss, model.pred], feed_dict={
#             model.x: dummy_x,
#             model.y: dummy_y,
#             model.lr: config.lr,
#             model.keep_prob: config.drop_out
#         })

#         print loss

