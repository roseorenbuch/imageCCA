#!/usr/bin/env python

import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf

import numpy as np
import os.path
import argparse
import multiprocessing
from tqdm import tqdm
import functools
import pandas as pd

import image
from utilsCAE import getBatch
from modelsCAE import modelCAE

        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('input', help="directory of images or tsv files for list of paths")
    parser.add_argument('-o', '--outdir', help="output directory", default="./", type=str)
    
    parser.add_argument('-B', '--batchEpoch', help="batch size per epoch", type=int, default=1024)
    parser.add_argument('-b', '--batch', help="batch size", type=int, default=256)
    
    
    parser.add_argument('-e', '--epochs', help="number of epochs to perform", type=int, default=8000)
    parser.add_argument('--start', help="starting epoch", type=int, default=0)
    
    parser.add_argument('-S', '--newSize', help="resize images before sampling", type=int, default=512)
    parser.add_argument('-s', '--sampSize', help="size of sampling window after resizing", type=int, default=128)
    
    parser.add_argument('--save', help="save every N epochs", type=int, default=50)
    
    parser.add_argument('-f','--filterSize', help="conv kernal size", type=int, default=5)
    parser.add_argument('-p','--poolSize', help="pooling size", type=int, default=2)
    
    parser.add_argument('-w', '--weights', help="pretrained weights", type=str, default=None)
    
    parser.add_argument('-t', '--threads', help="number of threads", type=int, default=1)
    parser.add_argument('-g', '--gpus', help="number of gpus", type=int, default=0)
    
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    
    parser.add_argument('--force', default=False, action='store_true')

    args = parser.parse_args()
    
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    
    if args.input.endswith('.tsv') or args.input.endswith('.txt'):
        allPaths = pd.read_csv(args.input,header=None)[0].to_list()
    else:
        allPaths = image.list_pictures(args.input)
        
    print('Found %d images' % len(allPaths))

    # build model
    print('Building model')
    cae = modelCAE(args.filterSize, args.poolSize, args.sampSize, args.gpus, args.weights)

    # train the model
    print('Training model')
    for i in tqdm(range(args.start, args.start + args.epochs)):
        #print('Epoch: %d of %d' % (args.start + i, args.start + args.epochs))
        imBatch = getBatch(args.batchEpoch, allPaths, args.sampSize, args.newSize, args.threads)
        cae.fit(imBatch, imBatch, epochs=1, verbose=args.verbose, batch_size=args.batch)
        if (i + 1) % args.save == 0:
            cae.save_weights(args.outdir + '/pretrained_' + str(i + 1) + '.h5', overwrite=args.force)
    print('Complete')
        
