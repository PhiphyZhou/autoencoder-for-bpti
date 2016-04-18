from __future__ import division
'''
Authors: Xiaozhou Zhou, Satya Prateek, Poorya Mianjy, Javad Fotouhi
Adapted from the code in https://github.com/arendu/Autoencoder-WordEmbeddings $$
'''
from sklearn.cross_validation import cross_val_score
import numpy as np
#from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn import preprocessing
from random import shuffle
import json
import pandas as pd


def main():
	#read in the labels
	labels = []
	with open("../sample_data/states.txt") as f:
		labels = f.readlines()
	labels = map(lambda x: int(x.lstrip().split(" ")[1]), labels)
	Y = np.asarray(labels)
	features = ["chi1", "chi2", "rmsd", "hbonds", "chi1_chi2", "chi1_chi2_r", "chi1_chi2_hbonds", "chi1_chi2_hbonds_r"] 
# 	clfs = [MultinomialNB(), KNeighborsClassifier(n_neighbors=3), RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)]
	clfs = [KNeighborsClassifier(n_neighbors=8)]
# 	clf_name = ["Multinomial NB", "KNN", "Random Forests"]
	clf_name = ["KNN"]
	folds = ["5"]
	arrs = {}
	dec_arrs = {}
	results = {}
	for fold in folds:
		for feature in features:
			for clf, cname in zip(clfs, clf_name):
				x_scale = preprocessing.MinMaxScaler()
				x_dec_scale = preprocessing.MinMaxScaler()
				X = None 
				X_dec = None
				if feature == "chi1_chi2":
					X = np.hstack((arrs["chi1"], arrs["chi2"]))
					X = x_scale.fit_transform(X)
					X_dec = np.hstack((dec_arrs["chi1"], dec_arrs["chi2"]))
					X_dec = x_dec_scale.fit_transform(X_dec)
				elif feature == "chi1_chi2_hbonds":
					X = np.hstack((arrs["chi1"], arrs["chi2"], arrs["hbonds"]))
					X = x_scale.fit_transform(X)
					X_dec = np.hstack((dec_arrs["chi1"], dec_arrs["chi2"], dec_arrs["hbonds"]))
					X_dec = x_dec_scale.fit_transform(X_dec)
				elif feature == "chi1_chi2_r":
					X = np.hstack((arrs["chi1"], arrs["chi2"], arrs["rmsd"]))
					X = x_scale.fit_transform(X)
				elif feature == "chi1_chi2_r":
					X = np.hstack((arrs["chi1"], arrs["chi2"], arrs["rmsd"]))
					X = x_scale.fit_transform(X)
				elif feature == "chi1_chi2_hbonds_r":
					X = np.hstack((arrs["chi1"], arrs["chi2"], arrs["hbonds"], arrs["rmsd"]))
					X = x_scale.fit_transform(X)
				elif (feature == "rmsd"):
					arrs[feature] = np.asmatrix(json.load(open("../sample_data/" + feature + "_labeled.json")))
					X = arrs[feature]
					X = x_scale.fit_transform(X)
				else:
					#read in the data
					arrs[feature] = np.asmatrix(json.load(open("../sample_data/" + feature + "_labeled.json")))
					dec_arrs[feature] = np.asmatrix(json.load(open("../output/" + feature + "_encoded.json")))
					X = arrs[feature]
					X = x_scale.fit_transform(X)
					X_dec = dec_arrs[feature]
					X_dec = x_dec_scale.fit_transform(X_dec)

				if X is not None:
					data_scores = cross_val_score(clf, X, Y, cv=int(fold))
					print("Raw Data Vect. Accuracy for %s using %s with %s folds: %0.2f (+/- %0.2f)" % (feature, cname , fold, data_scores.mean(), data_scores.std()))
					results[(fold, feature, cname, "raw")] = {"Mean":data_scores.mean(),"Std":data_scores.std()}
				if X_dec is not None:
					data_with_decoded_scores = cross_val_score(clf, X_dec, labels, cv=int(fold))
					print("Data + Representation Vect. Accuracy: for %s using %s with %s folds: %0.2f (+/- %0.2f)" % (
						feature, cname, fold, data_with_decoded_scores.mean(), data_with_decoded_scores.std()))
					results[(fold, feature, cname, "encoded")] = {"Mean":data_with_decoded_scores.mean(),"Std":data_with_decoded_scores.std()}
	results_df = pd.DataFrame(results)
	results_df.to_pickle("classifier_results.p")
	print "Finished"
if __name__ == '__main__':
	main()
