#!/usr/bin/env python

#import keras
import tensorflow as tf

#from keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from tensorflow.python.keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import multi_gpu_model
import numpy as np
import os.path
import image
import argparse
import multiprocessing
from tqdm import tqdm
import functools

# ---------------------------------------------------------------------------- #
# CAE models
# ---------------------------------------------------------------------------- #

def modelCAE(filterSize, poolSize, sampSize, gpus, weights=None):
    #strategy = tf.distribute.MirroredStrategy()
    #with strategy.scope(): 
    # initialize cae
    cae = Sequential()

    # convolution + pooling 1
    cae.add(Convolution2D(8, (filterSize, filterSize), input_shape=(3, sampSize, sampSize), padding='same'))
    cae.add(MaxPooling2D(pool_size=(poolSize, poolSize)))
    cae.add(Activation('relu'))

    # convolution + pooling 2
    cae.add(Convolution2D(16, (filterSize, filterSize), padding='same'))
    cae.add(MaxPooling2D(pool_size=(poolSize, poolSize)))
    cae.add(Activation('relu'))

    # convolution + pooling 3
    cae.add(Convolution2D(32, (filterSize, filterSize), padding='same'))
    cae.add(MaxPooling2D(pool_size=(poolSize, poolSize)))
    cae.add(Activation('relu'))

    # convolution + pooling 4
    cae.add(Convolution2D(64, (filterSize, filterSize), padding='same'))
    cae.add(MaxPooling2D(pool_size=(poolSize, poolSize)))
    cae.add(Activation('relu'))

    # convolution + pooling 5
    cae.add(Convolution2D(128, (filterSize, filterSize), padding='same'))
    cae.add(MaxPooling2D(pool_size=(poolSize, poolSize)))
    cae.add(Activation('relu'))

    # dense network
    cae.add(Flatten())
    cae.add(Dense(1024))
    cae.add(Activation('relu'))
    cae.add(Dense(128*4*4))
    cae.add(Activation('relu'))
    cae.add(Reshape((128, 4, 4)))
    cae.add(Activation('relu'))

    # unpooling + deconvolution 1
    cae.add(UpSampling2D(size=(poolSize, poolSize)))
    cae.add(Convolution2D(64, (filterSize, filterSize), padding='same'))
    cae.add(Activation('relu'))

    # unpooling + deconvolution 2
    cae.add(UpSampling2D(size=(poolSize, poolSize)))
    cae.add(Convolution2D(32, (filterSize, filterSize), padding='same'))
    cae.add(Activation('relu'))

    # unpooling + deconvolution 3
    cae.add(UpSampling2D(size=(poolSize, poolSize)))
    cae.add(Convolution2D(16, (filterSize, filterSize), padding='same'))
    cae.add(Activation('relu'))

    # unpooling + deconvolution 4
    cae.add(UpSampling2D(size=(poolSize, poolSize)))
    cae.add(Convolution2D(8, (filterSize, filterSize), padding='same'))
    cae.add(Activation('relu'))

    # final unpooling + deconvolution
    cae.add(UpSampling2D(size=(poolSize, poolSize)))
    cae.add(Convolution2D(3, (filterSize, filterSize),  padding='same'))
    cae.add(Activation('sigmoid'))  # ADDITION -DM

    # compile and load pretrained weights
    if gpus > 1:
        cae = multi_gpu_model(cae,gpus=gpus)
    cae.compile(loss='mse', optimizer=Adam(lr=0.0005, decay=1e-5))
    if weights:
        #print('loading pretrained weights')
        cae.load_weights(weights)
    
    return cae


def modelEncode(cae, filterSize, poolSize, sampSize, gpus):
    if gpus > 1:
        cae = cae.layers[-2]
    
    # initialize encoder
    encode = Sequential()
    encode.add(Convolution2D(8, (filterSize, filterSize),
        input_shape=(3, sampSize, sampSize), padding='same',  weights=cae.layers[0].get_weights()))
    encode.add(MaxPooling2D(pool_size=(poolSize, poolSize)))
    encode.add(Activation('relu'))
    encode.add(Convolution2D(16, (filterSize, filterSize), padding='same', weights=cae.layers[3].get_weights()))
    encode.add(MaxPooling2D(pool_size=(poolSize, poolSize)))
    encode.add(Activation('relu'))
    encode.add(Convolution2D(32, (filterSize, filterSize), padding='same', weights=cae.layers[6].get_weights()))
    encode.add(MaxPooling2D(pool_size=(poolSize, poolSize)))
    encode.add(Activation('relu'))
    encode.add(Convolution2D(64, (filterSize, filterSize), padding='same', weights=cae.layers[9].get_weights()))
    encode.add(MaxPooling2D(pool_size=(poolSize, poolSize)))
    encode.add(Activation('relu'))
    encode.add(Convolution2D(128, (filterSize, filterSize), padding='same', weights=cae.layers[12].get_weights()))
    encode.add(MaxPooling2D(pool_size=(poolSize, poolSize)))
    encode.add(Activation('relu'))
    encode.add(Flatten())
    encode.add(Dense(1024, weights=cae.layers[16].get_weights()))
    encode.add(Activation('relu'))
    
    if gpus > 1:
        encode = multi_gpu_model(encode, gpus = gpus)
        
    encode.compile(loss='mse', optimizer='adam')
    
    return encode


def modelDecode(cae, filterSize, poolSize, gpus):
    if gpus > 1:
        cae = cae.layers[-2]
        
    # initialize decoder
    decode = Sequential()
    decode.add(Dense(128*4*4, input_dim=(1024), weights=cae.layers[18].get_weights()))
    decode.add(Activation('relu'))
    decode.add(Reshape((128, 4, 4)))
    decode.add(Activation('relu'))
    decode.add(UpSampling2D(size=(poolSize, poolSize)))
    decode.add(Convolution2D(64, (filterSize, filterSize), padding='same', weights=cae.layers[23].get_weights()))
    decode.add(Activation('relu'))
    decode.add(UpSampling2D(size=(poolSize, poolSize)))
    decode.add(Convolution2D(32, (filterSize, filterSize), padding='same', weights=cae.layers[26].get_weights()))
    decode.add(Activation('relu'))
    decode.add(UpSampling2D(size=(poolSize, poolSize)))
    decode.add(Convolution2D(16, (filterSize, filterSize), padding='same', weights=cae.layers[29].get_weights()))
    decode.add(Activation('relu'))
    decode.add(UpSampling2D(size=(poolSize, poolSize)))
    decode.add(Convolution2D(8, (filterSize, filterSize), padding='same', weights=cae.layers[32].get_weights()))
    decode.add(Activation('relu'))
    decode.add(UpSampling2D(size=(poolSize, poolSize)))
    decode.add(Convolution2D(3, (filterSize, filterSize), padding='same', weights=cae.layers[35].get_weights()))
    decode.add(Activation('sigmoid'))
    
    if gpus > 1:
        decode = multi_gpu_model(decode, gpus = gpus)
    
    decode.compile(loss='mse', optimizer='adam')
    
    return decode

def modelClassifier(cae, filterSize, poolSize, gpus):
    if gpus > 1:
        cae = cae.layers[-2]
        
    cl = Sequential()
    cl.add(Convolution2D(8, (filterSize, filterSize),
        input_shape=(3, sampSize, sampSize), padding='same',  weights=cae.layers[0].get_weights()))
    cl.add(MaxPooling2D(pool_size=(poolSize, poolSize)))
    cl.add(Activation('relu'))
    cl.add(Convolution2D(16, (filterSize, filterSize), padding='same', weights=cae.layers[3].get_weights()))
    cl.add(MaxPooling2D(pool_size=(poolSize, poolSize)))
    cl.add(Activation('relu'))
    cl.add(Convolution2D(32, (filterSize, filterSize), padding='same', weights=cae.layers[6].get_weights()))
    cl.add(MaxPooling2D(pool_size=(poolSize, poolSize)))
    cl.add(Activation('relu'))
    cl.add(Convolution2D(64, (filterSize, filterSize), padding='same', weights=cae.layers[9].get_weights()))
    cl.add(MaxPooling2D(pool_size=(poolSize, poolSize)))
    cl.add(Activation('relu'))
    cl.add(Convolution2D(128, (filterSize, filterSize), padding='same', weights=cae.layers[12].get_weights()))
    cl.add(MaxPooling2D(pool_size=(poolSize, poolSize)))
    cl.add(Activation('relu'))
    cl.add(Flatten())
    cl.add(Dense(1024, weights=cae.layers[16].get_weights()))
    cl.add(Activation('relu'))
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)


    cl.add(Dense(100))
    cl.add(Dropout(0.5))
    cl.add(Activation('relu'))

    if nClasses == 2:
        cl.add(Dense(1))
        cl.add(Dropout(0.5))
        cl.add(Activation('sigmoid'))
        
        if gpus > 1:
            cl = multi_gpu_model(cl, gpus = gpus)
        
        cl.compile(loss='binary_crossentropy', optimizer=Adam(lr=5e-7, decay=1e-5))
    else:
        cl.add(Dense(nClasses))
        cl.add(Dropout(0.5))
        cl.add(Activation('softmax'))
        
        if gpus > 1:
            cl = multi_gpu_model(cl, gpus = gpus)
            
        cl.compile(loss='categorical_crossentropy', optimizer=Adam(lr=5e-7, decay=1e-5))
    return cl