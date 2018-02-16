from __future__ import print_function

import numpy as np
from time import time

from datasets import PairDatasetAVX


import argparse

from itertools import chain
import functools
import helper
from queue import Queue
from threading import Thread

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf

def buildNet(dataShape, labelShape, classCount):
    netGraph = tf.Graph()
    tf.reset_default_graph()
    with netGraph.as_default():
        inputData = tf.placeholder(
            tf.float32, shape=[None] + dataShape)
        inputLabels = tf.placeholder(tf.int32, shape=[None] + labelShape)
        trainPhase = tf.placeholder(tf.bool)

        inputData = inputData * 2 + inputData

        net = tf.layers.conv2d(
            inputData, filters=8, kernel_size=3, padding="same")
        net = tf.layers.batch_normalization(net, training=trainPhase)
        net = tf.nn.relu(net)
        net = tf.layers.max_pooling2d(net, 2, 2)

        net = tf.layers.conv2d(
            inputData, filters=16, kernel_size=3, padding="same")
        net = tf.layers.batch_normalization(net, training=trainPhase)
        net = tf.nn.relu(net)
        net = tf.layers.max_pooling2d(net, 2, 2)

        net = tf.layers.conv2d(
            inputData, filters=32, kernel_size=3, padding="same")
        net = tf.layers.batch_normalization(net, training=trainPhase)
        net = tf.nn.relu(net)
        net = tf.layers.max_pooling2d(net, 2, 2)

        net = tf.layers.flatten(net)

        net = tf.layers.dense(net, 1024)
        net = tf.layers.batch_normalization(net, training=trainPhase)
        net = tf.nn.relu(net)

        net = tf.layers.dense(net, 1024)
        net = tf.layers.batch_normalization(net, training=trainPhase)
        net = tf.nn.relu(net)

        net = tf.layers.dense(net, labelShape[0] * classCount)
        net = tf.reshape(net, [-1, labelShape[0], classCount])
        trnLoss = tf.losses.sparse_softmax_cross_entropy(
            inputLabels, net)

        optimizer = tf.train.AdamOptimizer(
            learning_rate=0.001)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = optimizer.minimize(trnLoss)

        prob = tf.nn.softmax(net)
        init = tf.global_variables_initializer()


    return prob, netGraph, trnLoss, optimizer, inputData, inputLabels, init, trainPhase


dataShape = [40, 140, 1]
labelCount = 8
maxLabel = 37
batchSize = 64

prob, netGraph, trnLoss, optimizer, inputData, inputLabels, init, trainPhase = buildNet(
    dataShape, [labelCount], maxLabel)

gpu_memory_usage = 0.5
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_usage

# Initialize the variables (i.e. assign their default value)


runSmall = '.small'
t1 = time()
print('Reading training data')
dataPath = '/home/ihradis/projects/2018-01-24_LP/data_REID/'
dataTrn = PairDatasetAVX(
    dataPath + 'REID_140.trn' + runSmall, dataPath + 'characters.txt', dataPath,
    resolution=(40, 140))
print('Reading training data DONE in ', time()-t1)

t1 = time()
print('Reading testing data')
dataTst = PairDatasetAVX(
    dataPath + 'REID_140.tst' + runSmall, dataPath + 'characters.txt', dataPath,
    resolution=(40, 140))
print('Reading testing data DONE in ', time() - t1)

print("TRN shape", *dataTrn.images.shape)
print("TST shape", *dataTst.images.shape)


with tf.Session(graph=netGraph, config=config) as sess:
    sess.run(init)
    trnLossAcc = 0
    for i in range(50000):
        data, labels = dataTrn.getBatch(batchSize)
        data = data[:,0,...,np.newaxis]
        _, loss = sess.run(
            [optimizer, trnLoss],
            feed_dict={inputData: data, inputLabels: labels, trainPhase: True })
        trnLossAcc += loss

        if i % 200 == 199:
            lossAcc = 0
            errAcc = 0
            itCount = 10
            for j in range(itCount):
                data, labels = dataTst.getBatch(100)
                data = data[:,0,...,np.newaxis]
                loss, outProb = sess.run(
                    [trnLoss, prob],
                    feed_dict={inputData: data, inputLabels: labels, trainPhase: False})
                lossAcc += loss

                labels = labels.reshape(-1)
                outProb = outProb.reshape(-1, outProb.shape[-1])
                idx = np.argmax(outProb, axis=1)
                errAcc +=  1 - np.average(idx == labels)
            print('TRAIN', i, trnLossAcc / 200)
            print('TEST', i, lossAcc / itCount, errAcc / itCount)
            trnLossAcc = 0

