'''
CS 600.615 Big Data

This script demonstrates serial autoencoder decoding.
See NpLayers.py for the autoencoder implementation.

Parameters:
[folder for input data] [folder for output] [folder for auto-encoder object] [feature type in number]

feature type number: 
0 - chi1, 1 - chi2, 2 - hbonds, 3 - rmsd

Authors: Xiaozhou Zhou, Satya Prateek, Poorya Mianjy, Javad Fotouhi
Adapted from the code in https://github.com/arendu/Autoencoder-WordEmbeddings $$
'''


import gzip, sys, itertools, time
import pdb
import json
import NpLayers as L
from scipy.optimize import fmin_l_bfgs_b
import numpy as np

features = ["chi1","chi2","hbonds","rmsd"]

def encode(f):
    decoding_data = sys.argv[1]+f+"_labeled.json"
    data_out = sys.argv[2]+f+"_encoded.json"
    ae = sys.argv[3]+f+".nn"

    #Load trained autoencoder
    autoencoder = L.load(ae)

    #load data
    data = json.load(open(decoding_data,"rb"))
    input_width = len(data[0])
    assert (input_width == autoencoder.topology[0])

    # Size of the embeddings.
    inside_width = autoencoder.topology[1]

    # rescale and reshape the data to be fed into the autoencoder   
    X = np.asarray(data)
    if int(sys.argv[4]) == 2:
        X_std = X/3.1
    else:
        X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    data = [np.reshape(x, (len(x), 1)) for x in X_std]
    
    #Encode data points (note: get_representation() returns a numpy array)
    encoded = []
    for d in data:
        a = autoencoder.get_representation(d)
        a = a.reshape(len(a)).tolist()
        encoded.append(a)
    
    json.dump(encoded, open(data_out,"wb"))

if __name__ == '__main__':
    '''
    parameters:
    [input folder] [output folder] [folder for autoencoder] 
    '''
    
#   for f in features:
#       encode(f)

    encode(features[int(sys.argv[4])])











