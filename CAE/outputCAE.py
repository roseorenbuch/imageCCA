#!/usr/bin/env python

import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import warnings
warnings.filterwarnings("ignore")
from sklearn.decomposition import PCA
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import offsetbox
import numpy as np
import os.path

import copy
import PIL
import PIL.ImageOps
import multiprocessing
import pickle
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import functools
import argparse
import pandas as pd


import image
from utilsCAE import getOne, getBatch, getOneStatic
from modelsCAE import modelCAE, modelEncode

# ---------------------------------------------------------------------------- #
# Helper functions
# ---------------------------------------------------------------------------- #

import subprocess, re

# Nvidia-smi GPU memory parsing.
# Tested on nvidia-smi 370.23

def run_command(cmd):
    """Run command, return output as string."""
    output = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
    return output.decode("ascii")

def list_available_gpus():
    """Returns list of available GPU ids."""
    output = run_command("nvidia-smi -L")
    # lines of the form GPU 0: TITAN X
    gpu_regex = re.compile(r"GPU (?P<gpu_id>\d+):")
    result = []
    for line in output.strip().split("\n"):
        m = gpu_regex.match(line)
        assert m, "Couldnt parse "+line
        result.append(int(m.group("gpu_id")))
    return result

def gpu_memory_map():
    """Returns map of GPU id to memory allocated on that GPU."""

    output = run_command("nvidia-smi")
    gpu_output = output[output.find("GPU Memory"):]
    # lines of the form
    # |    0      8734    C   python                                       11705MiB |
    memory_regex = re.compile(r"[|]\s+?(?P<gpu_id>\d+)\D+?(?P<pid>\d+).+[ ](?P<gpu_memory>\d+)MiB")
    rows = gpu_output.split("\n")
    result = {gpu_id: 0 for gpu_id in list_available_gpus()}
    for row in gpu_output.split("\n"):
        m = memory_regex.search(row)
        if not m:
            continue
        gpu_id = int(m.group("gpu_id"))
        gpu_memory = int(m.group("gpu_memory"))
        result[gpu_id] += gpu_memory
    return result

def pick_gpu_lowest_memory():
    """Returns GPU with the least allocated memory"""

    memory_gpu_map = [(memory, gpu_id) for (gpu_id, memory) in gpu_memory_map().items()]
    best_memory, best_gpu = sorted(memory_gpu_map)[0]
    return best_gpu

# ---------------------------------------------------------------------------- #

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('input', help="directory of images or tsv files for list of paths")
    parser.add_argument('--subset',type=str,default=None)
    parser.add_argument('-o', '--outdir', help="output directory", default="./", type=str)
    
    parser.add_argument('--encoding', help="repAll.pkl from previous run of outputCAE", default=None, type=str)
    
    parser.add_argument('-K', help="number of PCA components", type=int, default=100)
    
    parser.add_argument('-B', '--batchEpoch', help="encodings to generate per image", type=int, default=1000)
    parser.add_argument('-b', '--batch', help="batch size", type=int, default=100)
    
    
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
    
    parser.add_argument('--noExamples', help="don't generate example encodings", default=False, action='store_true')
    
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    
    parser.add_argument('--force', default=False, action='store_true')

    args = parser.parse_args()
    
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
        
    
    if args.input.endswith('.tsv') or args.input.endswith('.txt'):
        allPaths = np.loadtxt(args.input,dtype=str).tolist()
    else:
        allPaths = image.list_pictures(args.input)
        
  
    
    # Generate encodings
    if not args.encoding:
        # build CAE model
        def get_encoding(i):
            paths = split_paths[i]
            device = devices[i]
            
            tf.keras.backend.clear_session()
            os.environ["CUDA_VISIBLE_DEVICES"] = device

            cae = modelCAE(args.filterSize, args.poolSize, args.sampSize, 1, args.weights)

            # build encode model
            encode = modelEncode(cae, args.filterSize, args.poolSize, args.sampSize, 1)

            repAll = np.zeros((len(paths), 1024))
            if device == '0':
                for i in tqdm(range(len(repAll))):
                    for e in range(int(args.batchEpoch/args.batch)):
                        batch = getOne(args.batch, paths[i], args.sampSize, args.newSize, args.threads)
                    repAll[i, :] += sum(encode.predict(batch))
                    repAll[i, :] /= args.batchEpoch
            else:
                for i in range(len(repAll)):
                    for e in range(int(args.batchEpoch/args.batch)):
                        batch = getOne(args.batch, paths[i], args.sampSize, args.newSize, args.threads)
                    repAll[i, :] += sum(encode.predict(batch))
                    repAll[i, :] /= args.batchEpoch
            return repAll
        
        
        #split_paths = np.array_split(allPaths,2)
        #devices = ['0,1','2,3']
        #repAll = process_map(get_encoding, range(2), max_workers=2)

        #split_paths = np.array_split(allPaths,4)
        #devices = ['0','1','2','3']
        #repAll = process_map(get_encoding, range(4), max_workers=args.gpus)
        #repAll = np.asarray(repAll)
        
        
        
        
        cae = modelCAE(args.filterSize, args.poolSize, args.sampSize, args.gpus, args.weights)

        # build encode model
        encode = modelEncode(cae, args.filterSize, args.poolSize, args.sampSize, args.gpus)
        
        repAll = np.zeros((len(allPaths), 1024))
        for i in tqdm(range(len(repAll))):
            for e in range(int(args.batchEpoch/args.batch)):
                batch = getOne(args.batch, allPaths[i], args.sampSize, args.newSize, args.threads)
            repAll[i, :] += sum(encode.predict(batch))
            repAll[i, :] /= args.batchEpoch
        
            
        with open(''.join([args.outdir, 'repAll.pkl']) , 'wb') as file:
            pickle.dump(repAll, file)
       
    # Load encodings
    else:
        with open(args.encoding, 'rb') as file:
            repAll = pickle.load(file)

    # Generate encoding/decoding for some random images
    if not args.noExamples:
        
        #cae = modelCAE(args.filterSize, args.poolSize, args.sampSize, args.gpus, args.weights)
        cae = modelCAE(args.filterSize, args.poolSize, args.sampSize, 1, args.weights)
        
        exampleDir = ''.join([args.outdir,'examples/'])
        if not os.path.exists(exampleDir):
            os.makedirs(exampleDir)
        np.random.seed(0)
        if args.subset:
            imPaths = np.random.choice(subset,100)
        else:
            imPaths = np.random.choice(allPaths,100)
        ims = np.asarray([getOneStatic(imPath, args.sampSize, args.newSize) for imPath in imPaths])
        #ims = getBatch(101, allPaths, args.sampSize, args.newSize, args.threads)
        for i in range(100):
            image.array_to_img(ims[i] * 255.).save(''.join([exampleDir, str(i), 'A.png']))
            image.array_to_img(cae.predict(ims[i:i+1])[0] * 255.).save(''.join([exampleDir, str(i), 'B.png']))
            
    
    
    # Perform PCA on encodings
    if args.subset:
        subset = list(pd.read_csv(args.subset,header=None)[0])
        subset_indices = [i for i,j in enumerate(allPaths) if j in subset]
        repAll = repAll[subset_indices]
        
    
    pca = PCA(n_components=args.K, whiten=True)
    reduced = pca.fit_transform(repAll)
    vari = pca.explained_variance_ratio_
    # nDims = np.where(np.cumsum(vari) > .98)[0][0]
    # reduced = reduced[:,:nDims]
    nDims = reduced.shape[1]
    
    with open(''.join([args.outdir,'images.txt']), 'w') as fi:
        for x in allPaths:
            fi.write(x + '\n')

    with open(''.join([args.outdir,'vari.txt']), 'w') as fi:
        for x in np.cumsum(vari):
            fi.write(str(x) + '\n')

    fi = open(''.join([args.outdir, 'rep.csv']),'w')
    for i in range(len(repAll)):
        fi.write(os.path.basename(allPaths[i]) + ',')
        for dim in range(nDims):
            fi.write(str(repAll[i][dim]))
            if dim < nDims - 1:
                fi.write(',')
        fi.write('\n')
    fi.close()

    fi = open(''.join([args.outdir, 'repPCA.csv']),'w')
    for i in range(len(repAll)):
        fi.write(os.path.basename(allPaths[i]) + ',')
        for dim in range(nDims):
            fi.write(str(reduced[i][dim]))
            if dim < nDims - 1:
                fi.write(',')
        fi.write('\n')
    fi.close()