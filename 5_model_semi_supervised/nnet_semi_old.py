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

batch_size = 32

class Config():
    B, W, H, C = batch_size, 32,32, 9
    train_step = 25000
    lr = 2e-5
    weight_decay = 5e-5
    drop_out = 1
    sample_size = 10

    # load_path = '/atlas/u/jiaxuan/data/MODIS_data_county_processed_compressed/'
    load_path = "/atlas/u/jiaxuan/data/google_drive/img_output/"
    # load_path = "/atlas/u/jiaxuan/data/google_drive/img_full_output/"

    # save_path = '/atlas/u/jiaxuan/data/train_results/histogram_new/test21/'
    # save_path = '/atlas/u/jiaxuan/data/train_results/histogram_new/test22_optimize/'
    save_path = '/atlas/u/jiaxuan/data/train_results/final/monthly/'


def conv2d(input_data, out_channels, filter_size,stride, in_channels=None, name="conv2d"):
    if not in_channels:
        in_channels = input_data.get_shape()[-1]
    with tf.variable_scope(name):
        W = tf.get_variable("W", [filter_size, filter_size, in_channels, out_channels],
                initializer=tf.contrib.layers.variance_scaling_initializer())
        b = tf.get_variable("b", [1, 1, 1, out_channels])
        return tf.nn.conv2d(input_data, W, [1, stride, stride, 1], "SAME") + b

def deconv2d(input_data, out_channels, filter_size,stride, in_channels=None, name="conv2d"):
    if not in_channels:
        in_channels = input_data.get_shape()[-1]
    with tf.variable_scope(name):
        in_shape = input_data.get_shape().as_list()
        out_shape = [batch_size, in_shape[1]*stride, in_shape[2]*stride, out_channels]
        W = tf.get_variable("W", [filter_size, filter_size, out_channels, in_channels],
                initializer=tf.contrib.layers.variance_scaling_initializer())
        b = tf.get_variable("b", [1, 1, 1, out_channels])
        return tf.nn.conv2d_transpose(input_data, W, out_shape, [1, stride, stride, 1], padding="SAME") + b


def pool2d(input_data, ksize, name="pool2d"):
    with tf.variable_scope(name):
        return tf.nn.max_pool(input_data, [1, ksize, ksize, 1], [1, ksize, ksize, 1], "SAME")


def conv_batch_relu(input_data, out_channels, filter_size,stride, in_channels=None, name="crb"):
    with tf.variable_scope(name):
        a = conv2d(input_data, out_channels, filter_size, stride, in_channels)
        b = batch_normalization(a,axes=[0,1,2])
        r = tf.nn.relu(b)
        return r
def deconv_batch_relu(input_data, out_channels, filter_size,stride, in_channels=None, name="crb"):
    with tf.variable_scope(name):
        a = deconv2d(input_data, out_channels, filter_size, stride, in_channels)
        b = batch_normalization(a,axes=[0,1,2])
        r = tf.nn.relu(b)
        return r

def conv_batch_tanh(input_data, out_channels, filter_size,stride, in_channels=None, name="crb"):
    with tf.variable_scope(name):
        a = conv2d(input_data, out_channels, filter_size, stride, in_channels)
        b = batch_normalization(a,axes=[0,1,2])
        r = tf.nn.tanh(b)
        return r
def deconv_batch_tanh(input_data, out_channels, filter_size,stride, in_channels=None, name="crb"):
    with tf.variable_scope(name):
        a = deconv2d(input_data, out_channels, filter_size, stride, in_channels)
        b = batch_normalization(a,axes=[0,1,2])
        r = tf.nn.tanh(b)
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
        mu, variance = tf.nn.moments(input_data, axes, keep_dims=True, name="moments")
        return tf.nn.batch_normalization(input_data, mu, variance, None, None, 1e-6, name="batch")

def concat_tile(x,y):
    x_shape = x.get_shape().as_list()
    y = tf.reshape(y,[batch_size,1,1,1])
    y_tile = tf.tile(y,[1,x_shape[1],x_shape[2],1])
    return tf.concat(3,[x,y_tile])

def net_x2y(x,keep_prob = 1,name='net_x2y',reuse = False):
    with tf.variable_scope(name, reuse = reuse):
        conv1_1 = conv_batch_relu(x, 128, 3,1, name="conv1_1")
        conv1_1_d = tf.nn.dropout(conv1_1, keep_prob)
        conv1_2 = conv_batch_relu(conv1_1_d, 256, 3,2, name="conv1_2")
        conv1_2_d = tf.nn.dropout(conv1_2, keep_prob)

        conv2_1 = conv_batch_relu(conv1_2_d, 256, 3,1, name="conv2_1")
        conv2_1_d = tf.nn.dropout(conv2_1, keep_prob)
        conv2_2 = conv_batch_relu(conv2_1_d, 512, 3,2, name="conv2_2")
        conv2_2_d = tf.nn.dropout(conv2_2, keep_prob)

        conv3_1 = conv_batch_relu(conv2_2_d, 512, 3,1, name="conv3_1")
        conv3_1_d = tf.nn.dropout(conv3_1, keep_prob)
        conv3_2= conv_batch_tanh(conv3_1_d, 1024, 3,2, name="conv3_2")
        conv3_2_d = tf.nn.dropout(conv3_2, keep_prob)

        dim = np.prod(conv3_2_d.get_shape().as_list()[1:])
        flattened = tf.reshape(conv3_2_d, [-1, dim])

        # fc6 = dense(flattened, 1024, name="fc6")
        # fc6_tanh = tf.nn.tanh(fc6)

        y = tf.squeeze(dense(flattened, 2, name="dense"))
        y_lmu = y[:,0:1]
        y_lsgms = y[:,1:2]
        
        return (y_lmu, y_lsgms)


def net_xy2z(x,y,keep_prob = 1,name='net_xy2z',reuse = False):
    with tf.variable_scope(name,reuse = reuse):
        input_data = concat_tile(x,y)
        conv1_1 = conv_batch_relu(input_data, 128, 3,1, name="conv1_1")
        conv1_1_d = tf.nn.dropout(conv1_1, keep_prob)
        conv1_1_d = concat_tile(conv1_1_d, y)
        conv1_2 = conv_batch_relu(conv1_1_d, 256, 3,2, name="conv1_2")
        conv1_2_d = tf.nn.dropout(conv1_2, keep_prob)
        conv1_2_d = concat_tile(conv1_2_d, y)

        conv2_1 = conv_batch_relu(conv1_2_d, 256, 3,1, name="conv2_1")
        conv2_1_d = tf.nn.dropout(conv2_1, keep_prob)
        conv2_1_d = concat_tile(conv2_1_d, y)
        conv2_2 = conv_batch_relu(conv2_1_d, 512, 3,2, name="conv2_2")
        conv2_2_d = tf.nn.dropout(conv2_2, keep_prob)
        conv2_2_d = concat_tile(conv2_2_d, y)

        conv3_1 = conv_batch_relu(conv2_2_d, 512, 3,1, name="conv3_1")
        conv3_1_d = tf.nn.dropout(conv3_1, keep_prob)
        conv3_1_d = concat_tile(conv3_1_d, y)
        conv3_2= conv_batch_tanh(conv3_1_d, 1024, 3,2, name="conv3_2")
        conv3_2_d = tf.nn.dropout(conv3_2, keep_prob)
        conv3_2_d = concat_tile(conv3_2_d, y)

        dim = np.prod(conv3_2_d.get_shape().as_list()[1:])
        flattened = tf.reshape(conv3_2_d, [-1, dim])

        z = tf.squeeze(dense(flattened, 2048, name="fc6"))

        z_lmu = z[:,0:1024]
        z_lsgms = z[:,1024:2048]
        
        return (z_lmu, z_lsgms)

def net_yz2x(y,z,keep_prob = 1,name='net_yz2x',reuse = False):
    with tf.variable_scope(name,reuse = reuse):
        input_data = tf.concat(1,[z,y])
        fc1 = dense(input_data, 4*4*1024, name="fc1")
        fc1 = tf.reshape(fc1,[-1,4,4,1024])

        conv1_1 = deconv_batch_relu(fc1, 512, 3,2, name="conv1_1")
        conv1_1_d = tf.nn.dropout(conv1_1, keep_prob)
        conv1_1_d = concat_tile(conv1_1_d, y)
        conv1_2 = deconv_batch_relu(conv1_1_d, 512, 3,1, name="conv1_2")
        conv1_2_d = tf.nn.dropout(conv1_2, keep_prob)
        conv1_2_d = concat_tile(conv1_2_d, y)

        conv2_1 = deconv_batch_relu(conv1_2_d, 256, 3,2, name="conv2_1")
        conv2_1_d = tf.nn.dropout(conv2_1, keep_prob)
        conv2_1_d = concat_tile(conv2_1_d, y)
        conv2_2 = deconv_batch_relu(conv2_1_d, 256, 3,1, name="conv2_2")
        conv2_2_d = tf.nn.dropout(conv2_2, keep_prob)
        conv2_2_d = concat_tile(conv2_2_d, y)

        conv3_1 = deconv_batch_relu(conv2_2_d, 128, 3,2, name="conv3_1")
        conv3_1_d = tf.nn.dropout(conv3_1, keep_prob)
        conv3_1_d = concat_tile(conv3_1_d, y)
        conv3_2= deconv_batch_tanh(conv3_1_d, 18, 3,1, name="conv3_2")
        conv3_2_d = tf.nn.dropout(conv3_2, keep_prob)
        
        xx_lmu = conv3_2_d[:,:,:,0:9]
        xx_lsgms = conv3_2_d[:,:,:,9:18]

        return (xx_lmu,xx_lsgms)
def L_xy(x_mu,x_sgm,x,y,z_mu,z_lsgms):
    # log[p(x|y,z)]
    x_mu = tf.reshape(x_mu, [-1])
    x_sgm = tf.reshape(x_sgm, [-1])
    x = tf.reshape(x, [-1])
    log_x = tf.reduce_mean(tf.contrib.distributions.Normal(mu = x_mu,sigma = x_sgm).log_pdf(x))

    # log[p(y)]
    # y_mu_prior = 3.993
    # y_sgm_prior = 1.054
    y_mu_prior = 10
    y_sgm_prior = 10
    y_mu_prior = tf.reshape(tf.ones_like(y)*y_mu_prior,[-1])
    y_sigma_prior = tf.reshape(tf.ones_like(y)*y_sgm_prior,[-1])
    y = tf.reshape(y, [-1])
    log_y = tf.reduce_mean(tf.contrib.distributions.Normal(mu = y_mu_prior,sigma = y_sigma_prior).log_pdf(y))

    # log[p(z)] - log[q(z|x,y)]
    log_z = 0.5*tf.reduce_mean(1+z_lsgms-z_mu*2-tf.exp(z_lsgms))

    objective = -(log_x+log_y+log_z)
    return objective
    
def U_x(x_mu,x_sgm,x,z_mu,z_lsgms,y_mu_post,y_sgm_post,sample_size = 10):
    objective = 0
    for i in range(sample_size):
        # sample y from q(y|x)
        y_seed = tf.random_normal([batch_size])
        y_mu_post = tf.reshape(y_mu_post, [-1])
        y_sgm_post = tf.reshape(y_sgm_post, [-1])
        y = y_seed*y_sgm_post+y_mu_post
        y = tf.reshape(y, [-1])
        L = L_xy(x_mu, x_sgm, x, y, z_mu, z_lsgms)
        log_y = tf.reduce_mean(tf.contrib.distributions.Normal(mu = y_mu_post,sigma = y_sgm_post).log_pdf(y))
        objective += L-log_y
    objective /= sample_size
    return objective
def C_xy(y,y_mu_post,y_sgm_post):
    y_mu_post = tf.reshape(y_mu_post, [-1])
    y_sgm_post = tf.reshape(y_sgm_post, [-1])
    y = tf.reshape(y, [-1])
    log_y = tf.reduce_mean(tf.contrib.distributions.Normal(mu = y_mu_post,sigma = y_sgm_post).log_pdf(y))
    objective = -log_y
    return objective




class NeuralModel():
    def __init__(self, config, name):
        # hyperparameter settings
        self.lr = tf.placeholder(tf.float32, [])
        # get data
        self.x = tf.placeholder(tf.float32, [config.B*2, config.W, config.H, config.C])
        self.y_lab = tf.placeholder(tf.float32, [config.B, 1])
        self.x_lab = self.x[0:config.B,:,:,:]
        self.x_ulab = self.x[config.B:config.B*2,:,:,:]

        '''
        for labelled data
        '''
        # get y_stat
        self.y_lmu_lab, self.y_lsgms_lab = net_x2y(self.x_lab,reuse=False)
        self.y_mu_lab = tf.exp(self.y_lmu_lab)
        self.y_sgm_lab = tf.exp(self.y_lsgms_lab/2)
        # get z_stat
        self.z_lmu_lab, self.z_lsgms_lab = net_xy2z(self.x_lab, self.y_lab,reuse=False)
        self.z_mu_lab = tf.exp(self.z_lmu_lab)
        self.z_sgm_lab = tf.exp(self.z_lsgms_lab/2)
        # sample z
        self.z_seed_lab = tf.random_normal([config.B,1024])
        self.z_lab = self.z_seed_lab*self.z_sgm_lab+self.z_mu_lab
        # get xx_stat
        self.xx_lmu_lab, self.xx_lsgms_lab = net_yz2x(self.y_lab, self.z_lab,reuse=False)
        self.xx_mu_lab = tf.exp(self.xx_lmu_lab)
        self.xx_sgm_lab = tf.exp(self.xx_lsgms_lab/2)
        # # sample xx
        # self.xx_seed_lab = tf.random_normal([32,config.W, config.H, config.C])
        # self.xx_lab = self.xx_seed_lab*self.xx_sgm_lab+self.xx_mu_lab
        self.L = L_xy(self.xx_mu_lab, self.xx_sgm_lab, self.x_lab,
                        self.y_lab, self.z_mu_lab, self.z_lsgms_lab)
        alpha = 0.1*config.B*2
        self.C = C_xy(self.y_lab, self.y_mu_lab, self.y_sgm_lab)*alpha

        '''
        for unlabelled data
        '''
        # get y_stat
        self.y_lmu_ulab, self.y_lsgms_ulab = net_x2y(self.x_ulab, reuse = True)
        self.y_mu_ulab = tf.exp(self.y_lmu_ulab)
        self.y_sgm_ulab = tf.exp(self.y_lsgms_ulab/2)
        # sample y
        self.y_seed_ulab = tf.random_normal([config.B,1])
        self.y_ulab = self.y_seed_ulab*self.y_sgm_ulab+self.y_mu_ulab
        # get z_stat
        self.z_lmu_ulab, self.z_lsgms_ulab = net_xy2z(self.x_ulab, self.y_ulab, reuse = True)
        self.z_mu_ulab = tf.exp(self.z_lmu_ulab)
        self.z_sgm_ulab = tf.exp(self.z_lsgms_ulab/2)
        # sample z
        self.z_seed_ulab = tf.random_normal([config.B,1024])
        self.z_ulab = self.z_seed_ulab*self.z_sgm_ulab+self.z_mu_ulab
        # get xx_stat
        self.xx_lmu_ulab, self.xx_lsgms_ulab = net_yz2x(self.y_ulab, self.z_ulab, reuse = True)
        self.xx_mu_ulab = tf.exp(self.xx_lmu_ulab)
        self.xx_sgm_ulab = tf.exp(self.xx_lsgms_ulab/2)
        # # sample xx
        # self.xx_seed_ulab = tf.random_normal([32,config.W, config.H, config.C])
        # self.xx_ulab = self.xx_seed_ulab*self.xx_sgm_ulab+self.xx_mu_ulab
        self.U = U_x(self.xx_mu_ulab, self.xx_sgm_ulab, self.x_ulab,
                        self.z_mu_ulab, self.z_lsgms_ulab, self.y_mu_ulab, self.y_sgm_ulab,sample_size = config.sample_size)

        '''
        Train
        '''
        # weight decay (prior on weights)
        self.R = config.weight_decay*tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])

        self.loss = self.L+self.C+self.U+self.R
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        '''
        Predict
        '''
        self.pred = tf.reduce_mean(self.y_mu_lab)
        self.real = tf.reduce_mean(self.y_lab)
        self.predict_err = tf.nn.l2_loss(self.y_mu_lab-self.y_lab)

        

if __name__ == '__main__':
    config = Config()
    model= NeuralModel(config,'net')

    # Launch the graph.
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.initialize_all_variables())
    x = np.random.rand(config.B, config.W, config.H, config.C)
    x = np.concatenate((x, x),axis=0)
    y = np.random.rand(config.B, 1)*10
    # test_y = np.random.rand(config.B, 1)
    # test_z = np.random.rand(config.B, 1024)
    for i in range(500):
        pred_all = 0
        if i==50:
            config.lr *= 10
        for j in range(1):
            _,L,C,U,R,loss,pred,real, y_mu_lab,y_mu_ulab,y_sgm_lab,y_sgm_ulab = sess.run(
                [model.train_op,model.L,model.C,model.U,model.R,model.loss,model.pred,model.real,
                model.y_mu_lab,model.y_mu_ulab,model.y_sgm_lab,model.y_sgm_ulab], feed_dict={
                model.lr:config.lr,
                model.x:x,
                model.y_lab:y
                })
            pred_all += pred
        pred_all /= 1
        print 'step',i,'Lab',L,'\t','Class',C,'\t','Unlab',U,'\t','Reg',R,'\t','Total',loss
        print 'pred',pred_all,'real',real,'lr',config.lr
        # print y_mu_lab
        # print y
        # print y_mu_ulab-y
