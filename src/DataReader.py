''' 
This module is for preprocessing the bpti feature files.
Each features_#.txt file is considered as a high dimensional data point, which has 4 types of features: chi1, chi2, hbonds, rmsd. Each of the 4 types of feature arrays of the snapshots is concatenated into one big array. Then the 4 big arrays are serialized for further use. 

Usage: 
python DataReader.py <the folder of all the feature files/> <output folder/> [starting index of data file] [ending index of data file]

Authors: Xiaozhou Zhou, Satya Prateek, Poorya Mianjy, Javad Fotouhi

'''

import json 
import sys, os

# number of frames in one block
block_size = 10 

# step size of reading the feature files
step = 250

def preprocess(start, end, input_folder):

	chi1_all = []
	chi2_all = []
	hbonds_all = []
	rmsd_all = []

	# deal with each feature file
	for i in xrange(int(start),int(end),step):
		print "Processing file "+`i`
		file_name = input_folder+"features_"+`i`+".txt"
		file = open(file_name,'r')
		raw_str = file.read() # the txt file is a single line
		json_str = raw_str.replace("'",'"') # change to standard JSON format
		data_all = json.loads(json_str) # create a list of dicts.  
		# print type(data_all[0]['hbonds'][3])
		# print len(data_all)

		# Deal with windows with 10 frames each
		for s in xrange(0,len(data_all),block_size):
		
			# concatenate the same type of features together
			chi1 = []
			chi2 = []
			hbonds = []
			rmsd = []

			for f in data_all[s:s+block_size]:
				chi1 += f['chi1']	
				chi2 += f['chi2']
				hbonds += f['hbonds']
				rmsd += f['rmsd']
	
				chi1_all.append(chi1)
				chi2_all.append(chi2)
				hbonds_all.append(hbonds)
				rmsd_all.append(rmsd)
	
	return chi1_all, chi2_all, hbonds_all, rmsd_all


def store_all(input_folder,output_folder,start,end):
	'''
	serialize each feature array into JSON file and store them in the current directory
	'''

	chi1_all, chi2_all, hbonds_all, rmsd_all = preprocess(start,end,input_folder)
	print "Writing json files..."
	with open(output_folder+'chi1.json', 'w') as c1, open(output_folder+'chi2.json', 'w') as c2, open(output_folder+'hbonds.json', 'w') as h,open(output_folder+'rmsd.json', 'w') as r:
		print "Writing chi1.json..."
		json.dump(chi1_all, c1)
		print "Writing chi2.json..."
		json.dump(chi2_all, c2)
		print "Writing hbonds.json..."
		json.dump(hbonds_all, h)
		print "Writing rmsd.json..."
		json.dump(rmsd_all, r)

	# test the result
	# a = json.load(open(output_folder+"chi1.json","r"))
	# print len(a)
	# print len(a[0])


def store_labeled(input_folder,output_folder,start,end):
	''' 
	only store the middle block in each file which has the label
	'''
	chi1_all, chi2_all, hbonds_all, rmsd_all = preprocess(start,end,input_folder)
	file_num = int(end) - int(start)
	chi1 = []
	chi2 = []
	hbonds = []
	rmsd = []
	
	for i in xrange(500/block_size+1,file_num*1000/block_size,1000/block_size):
		chi1.append(chi1_all[i])
		chi2.append(chi2_all[i])
		hbonds.append(hbonds_all[i])
		rmsd.append(rmsd_all[i])
	with open(output_folder+'chi1_labeled.json', 'w') as c1, open(output_folder+'chi2_labeled.json', 'w') as c2, open(output_folder+'hbonds_labeled.json', 'w') as h,open(output_folder+'rmsd_labeled.json', 'w') as r:
		print "Writing chi1_labeled.json..."
		json.dump(chi1, c1)
		print "Writing chi2_labeled.json..."
		json.dump(chi2, c2)
		print "Writing hbonds_labeled.json..."
		json.dump(hbonds, h)
		print "Writing rmsd_labeled.json..."
		json.dump(rmsd, r)
		
	
if __name__ == '__main__':
	input_folder = sys.argv[1]
	output_folder = sys.argv[2]
	start = sys.argv[3]
	end = sys.argv[4]
	
 	store_all(input_folder,output_folder,start,end)
#	store_labeled(input_folder,output_folder,start,end)





