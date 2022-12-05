import pandas as pd 
import numpy as np
from hash import get_hash_list, hash_function
from functions import add_to_dict

sig_matrix = pd.read_csv('Signature_matrix.csv')
sig_matrix = sig_matrix.drop('Unnamed: 0', axis=1).to_numpy()

n = len(sig_matrix)
print(n)
min_bands = 2
max_bands = int(round(n*0.5))

#Outputs a list with possible candidate tuples (b,r) for (#bands, #rows) for which b * r = n
def get_bandrow(sig_matrix, min_bands, max_bands):
    n = len(sig_matrix)
    bandrow_list = []
    
    for bands in range(min_bands, max_bands):
        if n % bands == 0:
            r = n/bands
            bandrow_list.append((bands,r))

#b_r_candidate = get_bandrow(sig_matrix, min_bands, max_bands)
#print(b_r_candidate)


#LSH
def LSH(sig_matrix, bands):
    #Check if b * r = n
    n = len(sig_matrix)
    m = len(sig_matrix[0])
    assert (n%bands == 0)

    #Initialization of row index
    r = int(n/bands)
    lwrbound_index = 0
    upprbound_index = r
    LSH_buckets = {}
    for b in range(bands):
        #Get different hash-function for every band (sig_size determines number of hash functions and num_elem_univer_set influences
        #the number of bins used).
        hash_table = get_hash_list(sig_size= bands, num_elem_univer_set= n)
        print(hash_table)

        for product in range(m):                        #For every product
            hash_key = ""
            for index in range(lwrbound_index, upprbound_index):            #upprbound not included!
                hash_key = hash_key + str(sig_matrix[index][product])

            #Hash the concatenated value to a bucket
            hash_value = hash_function(hash_table[b][0], hash_table[b][1], hash_table[b][2], int(hash_key))
            # Store the hash value in a dictionary or so.
            add_to_dict(LSH_buckets, hash_value, product)          #Store product in bucket 
        #update index values
        lwrbound_index = lwrbound_index + r
        upprbound_index = upprbound_index + r
 




LSH(sig_matrix, 2)
