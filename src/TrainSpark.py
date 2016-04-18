'''
CS 600.615 Big Data

This script demonstrates parallel autoencoder training in Pyspark
using a simple averaging method.
See NpLayers.py for the autoencoder implementation.

Authors: Xiaozhou Zhou, Satya Prateek, Poorya Mianjy, Javad Fotouhi
Adapted from the code in https://github.com/arendu/Autoencoder-WordEmbeddings $$

parameters:
[input data folder/] [number of threads] [output folder/] [number of hidden nodes] [index of feature type]

Usage example:
/software/spark-1.2.1/bin/spark-submit --py-files src/NpLayers.py,src/utils.py src/TrainSpark.py /data/features/ 64 /output/ 50 0

'''

import gzip, sys, itertools, time
import pdb
from pyspark import SparkContext, SparkConf
import json

import NpLayers as L
from TrainSerial import make_data_from_file as make_data
from scipy.optimize import fmin_l_bfgs_b
import numpy as np

features = ["chi1","chi2","hbonds","rmsd"] 

# Train the parameters of the autoencoder in bae for a maximum of max_itr
# function evaluations of l-BFGS on some partition of data in itr. Returns
# partially optimized weights.
def train(itr, bae, max_itr=1):
    ae = bae.value
    xs = []
    for x in itr:
      xs.append(x)
    (xopt, fopt, return_status) = fmin_l_bfgs_b(ae.get_cost, 
                                                ae.get_network_weights(), 
                                                ae.get_gradient, args=(xs, ),
                                                pgtol=0.1, maxfun=max_itr)
    return 1, xopt


if __name__ == '__main__':
    
    # feature files you want to deal with
    start = 0
    end = 4000
    input_folder = sys.argv[1]
    num_chunks = float(sys.argv[2])
    output_folder = sys.argv[3]

    # Size of the embeddings.
    inside_width = int(sys.argv[4])
    
    # type of feature 0 - chi1, 1 - chi2, 2 - hbonds, 3 - rmsd
    feature_type = int(sys.argv[5])

    log = open(output_folder+features[feature_type]+"_"+str(inside_width)+"_spark_logs", "wb")
    
    reading_time = time.time()
    print >>log, 'making data...' 
    data, input_width = make_data(int(feature_type),input_folder,start,end)
    reading_time = time.time() - reading_time
    print >> log, 'time for reading data: '+str(reading_time)

    start_time = time.time()
    # This is the model we care about. The weights of this model will be updated to reflect the
    # average of multiple autoencoders (with identical topology) which are trained on different
    # subsets of the the training data.
    ae = L.Network(0.1, [input_width, inside_width, input_width])

    # Initial cost.
    prev_cost = ae.get_cost(ae.get_network_weights(), data)

    conf = (SparkConf()
         .setAppName("My app")
         .set("spark.executor.memory", "8g")
         .set("spark.storage.memoryFraction", 0.2))
    sc = SparkContext(conf=conf)

    rdd = sc.parallelize(data, numSlices=int(num_chunks))
    weights = None
    threshold = 3
    converged = False
    itr_converged = -1
    
    # Convergence usually occurs before 10 iterations. This should be increased
    # if input_width and inside_width are increased substantially. 
    itr = 0
    while itr < 10:
        cost = 0.0
        bae = sc.broadcast(ae)
        
        # Apply the map to the partitions of the training data
        weights = rdd.mapPartitions(lambda x: train(x, bae)).collect()

        # Sum up the weights that result from collect(). We sum up the 
        # the weights using the collect() from above and then stepping
        # through the results and adding up the weights. This appears to
        # be more stable than using a reduce which does the same thing,
        # which is why we use it here.
        summed_weights = weights[1]
        for i in range(2, len(weights)):
          # The even elements are the '1' labels, and the odd elements are the
          # weights that we want.
          if i % 2 == 0:
            continue
          summed_weights = np.add(summed_weights, weights[i])

        # Get the averaged weights and update the autoencoder.
        new_weights = summed_weights / num_chunks
        ae.set_network_weights(new_weights)
        cost = ae.get_cost(new_weights, data)
        elapsed_time = time.time() - start_time
        print  >>log, "Iteration", itr, "Cost:", cost, "Prev Cost:", prev_cost, "Elapsed time (s):", elapsed_time

        if (abs(cost - prev_cost) < threshold) and not converged:
            itr_converged = itr
            converged = True
        if converged:
            print  >>log, "Training has converged at iteration", itr_converged, ". Elapsed time (s):",  elapsed_time
        prev_cost = cost
        itr += 1
    L.dump(ae, output_folder+features[feature_type]+"_"+str(inside_width)+"_spark.nn")
    log.close()
