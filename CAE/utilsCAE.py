#!/usr/bin/env python

import numpy as np
import os.path
import image
import argparse
import multiprocessing
from tqdm import tqdm
import functools

# ---------------------------------------------------------------------------- #
# Helper functions
# ---------------------------------------------------------------------------- #

def getImage(sampSize, newSize, imPath, _=''):
    ''' randomly samples an image  '''
    f1 = int(np.sign(np.random.rand() - .5))
    f2 = int(np.sign(np.random.rand() - .5))
    im = image.load_img(imPath).resize((newSize, newSize))
    r  = int(np.random.rand() * (newSize - sampSize))
    c  = int(np.random.rand() * (newSize - sampSize))
    im = im.crop((r, c, r + sampSize, c + sampSize))
    im = image.random_rotation(im, 5)
    im = image.img_to_array(im)[:, ::-f1, ::-f2]
    im = (im - im.min())/(im.max() - im.min())
    return im

def getBatch(batchSize, allPaths, sampSize, newSize, threads):
    ''' helper function to get a random sampling from an image directory '''
    #allIms = np.zeros((batchSize, 3, sampSize, sampSize))
    
    imPaths = np.random.choice(allPaths, batchSize, replace = False)
    pool = multiprocessing.Pool(processes=threads)
    allIms = pool.map(functools.partial(getImage, sampSize, newSize), imPaths)
    pool.close() 
    allIms = np.asarray(allIms)
    return allIms/255.

def getBatchClassify(batchSize, allPaths, labels, sampSize, newSize, threads, numClass):
    imIndices = np.random.choose(range(len(allPaths)), batchSize)
    Y = np.asarray([[labels[i]] for i in imIndices]).astype('int')
    
    pool = multiprocessing.Pool(processes=threads)
    allIms = pool.map(functools.partial(getImage, sampSize, newSize, allPaths), range(batchSize))
    pool.close() 
    allIms = np.asarray(allIms)
    
    Y = np.zeros((batchSize, 1)).astype('int')
    if numClass > 2:
        Y2 = np.zeros((batchSize, numClass)).astype('int')
        for i in xrange(batchSize):
            Y2[i, Y[i][0]] = 1
        Y = Y2
    return allIms, Y

def getOne(batchSize, imPath, sampSize, newSize, threads):
    ''' helper function to get a random sampling from an image '''
    #allIms = np.zeros((batchSize, 3, sampSize, sampSize))
    #for i in range(batchSize):
    #    allIms[i,:] = getImage(sampSize, newSize, imPath)
    pool = multiprocessing.Pool(processes=threads)
    allIms = pool.map(functools.partial(getImage, sampSize, newSize, imPath), range(batchSize))
    pool.close() 
    allIms = np.asarray(allIms)
    return allIms/255.

def getOneStatic(imPath, sampSize, newSize):
    ''' randomly samples an image with no transformations  '''
    im = image.load_img(imPath).resize((newSize, newSize))
    r  = int(np.random.rand() * (newSize - sampSize))
    c  = int(np.random.rand() * (newSize - sampSize))
    im = im.crop((r, c, r + sampSize, c + sampSize))
    return image.img_to_array(im)/255.
