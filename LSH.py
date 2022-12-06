import pandas as pd 
import numpy as np
from hash import get_hash_list, hash_function
from functions import add_to_dict, add_to_set

sig_matrix = pd.read_csv('Signature_matrix.csv')
sig_matrix = sig_matrix.drop('Unnamed: 0', axis=1).to_numpy()

n = len(sig_matrix)
m = len(sig_matrix[0])
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
    return bandrow_list

#b_r_candidate = get_bandrow(sig_matrix, min_bands, max_bands)
#print(b_r_candidate)


#LSH function returns a set of tuples involving the products that are candidates.
#Out of all the possible band choices (s.t. b * r = n), at least 25 bands have to be chosen, for 20 it crashes as concatenating 
#75 numbers will be too large to convert back to an integer
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
       
        for product in range(m):                        #For every product
            hash_key = ""
            for index in range(lwrbound_index, upprbound_index):            #upprbound not included!
                hash_key = hash_key + str(int(sig_matrix[index][product]))

            #Hash the concatenated value to a bucket
            hash_value = hash_function(hash_table[b][0], hash_table[b][1], hash_table[b][2], int(hash_key))
            # Store the hash value in a dictionary or so.
            add_to_dict(LSH_buckets, hash_value, product)          #Store product in bucket 
        #update index values
        lwrbound_index = lwrbound_index + r
        upprbound_index = upprbound_index + r

    #Add pairs to consider to a set, to make sure that they are not checked multiple times.
    candidate_set = set()
    for key in LSH_buckets.keys():
        #Get candidate pairs:
        #only have to check when there are multiple products in same bucket:
        candidates = LSH_buckets[key]
        if type(candidates) == list:
            for el in candidates:
                for el2 in candidates:
                    if el!=el2:
                        #get every pair and add (x,y) tuple
                        #Possibly here already make a check to not include both (x,y) and (y,x)
                        add_to_set(candidate_set, (el,el2))

    return candidate_set
 

candidate_pairs = LSH(sig_matrix, 25)

#Idea for clustering: perhaps take the number of times products are hashed to each other into account, now considers only cases >=1
#Not efficient as it takes the full matrix instead of just the upper (bottom) half:
def get_distance_matrix(candidate_pairs, signature_matrix):
    n = len(sig_matrix)
    m = len(sig_matrix[0])
    distance_matrix = np.ones(shape=(m, m)) * np.inf

    for candidate in candidate_pairs:
        cand_1 = candidate[0]
        cand_2 = candidate[1]

        #Extract columns of products in signature matrix:
        sig_1 = signature_matrix[:,cand_1]
        sig_2 = signature_matrix[:,cand_2]

        #Get similarity between sig_1 and sig_2
        similarity_value = 2       #---> this should be the similarity measure, take inverse to be consistent

        #update (inversed similarity) in distance matrix (note the symmetry):
        distance_matrix[cand_1,cand_2] = similarity_value
        distance_matrix[cand_2,cand_1] = similarity_value
    return distance_matrix

#print(distance_matrix)


