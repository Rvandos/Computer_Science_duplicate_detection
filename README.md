# Product duplicate detection
This repository contains the code for the "Computer Science for Business Analytics (FEM21037)" assignment.
A scalable solution to the product duplicate detection problem has been implemented, which includes min-hashing, locality-sensitive hashing (LSH) and agglomerative
Hierarchical clustering.

This repository contains in total 5 python files: "Bootstrap.py", "functions.py", "hash.py", "performance.py", "plots.py". And three folders with remaining documents.

## Python files
### 1) "functions.py"
-This is the general python file, which contains all the functions used. These include functions for data pre-processing, obtaining binary and signature matrices, LSH, clustering, and functions to obtain performance measurements. All the functions in the file contain a small description, for this reason further information is left out.


### 2) "Bootstrap.py"
-This file makes use of the functions in "functions.py" and creates intermediary input files. It splits the data based on a bootstrapping approach into training and test data, which are then
seperately stored as json files in the "bootstrap_input" folder. Additionally, the signature matrices for both the training and test sets are computed and stored in the same folder.
This is done for convenience as it takes quite some time to obtain the signature matrices for 5 bootstrap samples.


### 3) "hash.py"
-This file is used to create a pre-specified number of distinct hash functions, where the hash functions take the following form: (a*x+b) mod(c). Here a and b are chosen to be distinct random integers up to some pre-specified number k. Then, c is chosen to be the next largest prime of k.


### 4) "performance.py"
-This file uses both the intermediary input files created by "Bootstrap.py" and functions from "functions.py". It computes the cluster threshold based on the training data set, and it performs LSH and clustering on the test set to obtain clusters containing duplicate pairs. Moreover, performance measures are created wit h






