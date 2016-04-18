Course Project for cs600.615
============================

Contributers:
----------------------------
Javad Fotouhi

Poorya Mianjy

Satya Prateek (sbommar1@jhu.edu)

Xiaozhou Zhou (xzhou18@jhu.edu)

### Warning: Don't run programs in the damsl gateway, instead, ssh to a working machine. Please see the file damsl-cluster.txt for all available machines. 

Using Docker to run a pyspark test script: 
----------------------------

ssh to one host machine where the image damsl/spark is stored. 

 ```
 ssh qp-hd4
 ```  
 
Create a docker container with the Spark image damsl/spark:

 ```
 docker run -it --name xpjs -v /damsl/projects/MD/Simulations/bpti:/data -v /home/bdslss15-xpjs/test:/test -v /output damsl/spark
 ```
 
 Note: You only need to create it once, or you can try to create other containers with a different name. 
 Volumes mounted:
 
   - /data: a read-only volume mounted from the bpti data folder on the host server, which includes all bpti data we need for this project
   
   - /test: a read-only volume mounted from the host server which contains some test spark script and data
   
   - /output: a read-write volume that can be used to hold the output
   
 After creating the container, use CTRL+P+Q to exit so that the container keeps running
 
Enter the bash shell of the container and go to the root directory:

 ```
 docker exec -it xpjs bash 
 cd ..
 ```
 
test spark using test script. (make sure that triangles.out is not in /output)

 ```
 /software/spark-1.2.1/bin/spark-submit test/triangle_count.py test/testdata
 ```
 
 Then you should see triangles.out in the /output folder
 
exit the container

 ```
 exit
 ```
 
 Alternatively you can use CTRL+P+Q
 
 
 
Preprocessing the feature files: 
----------------------------
In the root directory of the repo: 
```
python src/DataReader.py /damsl/projects/MD/Simulations/bpti/features/ data/
```
Please see the document in DataReader.py for more information. 
The output /data folder is too large (>10G) so DON'T push it into this repo. (already added "/data" to .gitignore) 
It's expected to cost more than an hour to preprocessing all feature files in non-spliting mode, and even longer time in splitting mode. 

If the json file is too large to open, you can check the data in python shell. For example:
```
python
>>> import json
>>> a = json.loads(open("data/chi1.json","r").read()) # parse the json file into an array
>>> print a[3] # print the feature values of the 3rd data point
```
The output is a big 1-D array.



