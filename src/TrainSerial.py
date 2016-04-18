'''
CS 600.615 Big Data

This script demonstrates serial autoencoder.
Authors: Xiaozhou Zhou, Satya Prateek, Poorya Mianjy, Javad Fotouhi
Adapted from the code in https://github.com/arendu/Autoencoder-WordEmbeddings $$

parameters: 
/data/features/ 0 4000 /output/ 2 200
[feature_file_folder/] [start_index] [end_index(+1)] [output_folder] [featuretype(integer)] [number of hidden nodes]

'''

import gzip
import pdb
import json,pickle
import NpLayers as L
import numpy as np
import utils
from os import listdir
from os.path import isfile, join
import sys
import DataReader as dr
import time


features = ["chi1","chi2","hbonds","rmsd"] 

def make_data_from_json():
    """
    Read data from the premade json files
    Prepare the features inputted to 
    NpLayers.Network() 
    feature_type = "chi1, chi2, hbonds,rmsd"
    sample = True/False
    Returns: Feature vector and number of features
    """
    feature_len = 0
    data = []
    file_name = "sample_data/hbonds_labeled.json"
#     sample_path = "/project/data/states.txt"
#     with open(sample_path) as sample_file:
#         samples = sample_file.readlines()
#     files_list = [join(data_path,f) for f in listdir(data_path) if isfile(join(data_path, f))] 
#     for file_name in files_list:
    with open(file_name, "r") as f:
        a = json.load(f)
        if feature_len == 0:
            feature_len = len(a[0])
#    print a[0]
    for b in a:
        b = np.asarray(b)
        data.append((b, b))
   # Scale the data to lie between 0 and 1
    X = np.asarray(data)
#     X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    X_std = X/3.1
#    X_std = (X + 180) / 360
    # rescale to make the angles lie in [0,1], then rescale again according to the length of features to prevent overflow in exponential
#    X_std = (X + 180) / 360/(feature_len/100)
    data = [(np.reshape(x, (len(x), 1)),  np.reshape(y, (len(y), 1))) for x, y in X_std]
#     pickle.dump(data,open("/output/"+features[feature_type]+"array","wb"))
    return data, feature_len

def make_data_from_file(feature_type,input_folder,start,end):
    """
    Read data from original feature files directly
    feature_type should be a number:
    0 - chi1, 1 - chi2, 2 - hbonds, 3 - rmsd
    """
        
    data = []
#   data = numpy.array([])
    feature_len = 0
    all_data = dr.preprocess(start, end, input_folder)
    
    for f in all_data[feature_type]:
        f = np.asarray(f)
        data.append((f,f))
        if feature_len == 0:
            feature_len = len(f)
    
    all_data = [] # release the memory
    X = np.asarray(data)
    if feature_type == 2:
    # special treatment for hbonds [0,3]
        X_std = X/3.1
    else:
        X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    data = [(np.reshape(x, (len(x), 1)),  np.reshape(y, (len(y), 1))) for x, y in X_std]
#   pickle.dump(data,open("/output/"+features[feature_type]+"array","wb"))
    return data, feature_len
    

if __name__ == '__main__':
    '''
    arguments: 
    1. path of the directory containing feature files
    2. starting index of the files
    3. ending index +1 of the files
    4. folder to store the autoencoder binary file
    5. integer for feature type to be trained: 
        0 - chi1, 1 - chi2, 2 - hbonds, 3 - rmsd
    6. number of nodes in the hidden layer
    '''
    feature_type = int(sys.argv[5])
    hidden_nodes = int(sys.argv[6]) # number of hidden nodes
        
    #reading data
    reading_time = time.time()
    output_folder = sys.argv[4]
    SAVE_TRAINED_NN = output_folder+features[feature_type]+".nn"  # give a better name here
    print 'Reading the sample corpus for feature {}.'.format(features[feature_type])
    
    # read data for chi1 and chi2
    data, feature_len = make_data_from_file(feature_type, sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
    
    # read data for hbonds (only central block of each file)
#     data, feature_len = make_data_from_json()
    
    print "reading data takes time: "+str(time.time()-reading_time)
    print "feature length: "+str(feature_len)
    
    # training autoencoder
    training_time = time.time()
    autoencoder = L.Network(0, [feature_len, hidden_nodes, feature_len], data)
    init_weights = autoencoder.get_network_weights()
    init_cost = autoencoder.get_cost(init_weights, data)
    print "weight length: "+str(len(init_weights))

#   final_weights = autoencoder.train_L(data)
#   final_cost = autoencoder.get_cost(final_weights, data)
#   print 'cost before training', init_cost, ' after L-BFGS-B training:', final_cost
#   training_time = time.time()-training_time
#   print "training takes time: "+str(training_time)

    final_weights = autoencoder.train_sgd(data)
    final_cost_sgd = autoencoder.get_cost(final_weights, data)
    print 'cost before training', init_cost, 'after sgd training', final_cost_sgd
    training_time = time.time()-training_time
    print "training takes time: "+str(training_time)
    
    autoencoder.set_network_weights(final_weights)
    L.dump(autoencoder, SAVE_TRAINED_NN)
