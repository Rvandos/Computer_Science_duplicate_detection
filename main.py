import json
import pandas as pd
import numpy as np
import regex as re
from hash import hash_function, get_hash_list

with open('TVs-all-merged.json') as f:
    data = json.load(f)


#get all elements  
def pre_process(data):
    df = {}
    mw_title = set()
    mw_value = set()
    for key in data.keys():
        #Add pre-processed data to new df
        df[key] = []

        #Pre-process title:
        key_title = data[key][0]['title']
        model_title = re.search("[a-zA-Z0-9]*(([0-9]+[ˆ0-9, ]+)|([ˆ0-9, ]+[0-9]+))[a-zA-Z0-9]*", key_title)
        model_title = model_title.group().replace(" ", "")
        df[key].append(model_title)
        #Add title of product to set object.
        mw_title.add(model_title)

        #Store website for non-duplicate detection:
        website = data[key][0]['shop']
        #Above does not do anything yet

        #pre-process key/value pairs:
        characteristics = data[key][0]['featuresMap']
        for items in characteristics.items():
            value = re.search("[a-zA-Z0-9]*(([0-9]+[ˆ0-9, ]+)|([ˆ0-9, ]+[0-9]+))[a-zA-Z0-9]*", items[1])
            #Now if regex does not find a match it does not add the value of the key-value pair!!
            if value != None:
                mw_value.add(value.group())
                df[key].append(value.group())
    return(mw_title, mw_value, df)

mw_title, mw_value, processed_data = pre_process(data)
#print(processed_data)



#Binary matrix:
mw = set.union(mw_title, mw_value)
n = len(mw)
m = len(processed_data.keys())
bin_matrix = np.zeros((n,m))

#create binary matrix
m=0
for key in processed_data.keys():
    k=0
    for model_word in mw:
        for item in processed_data[key]:
            if(model_word == item):
                bin_matrix[k, m] = 1
        k = k+1
    m = m+1


#print(bin_matrix)

#Minhashing function 
#Make signature size a hyper parameter to tune
signature_size = round(n * 0.5)                         #Reduce number of elements by half
minhash_matrix = np.ones(shape=(signature_size, m)) * np.inf

#Get hash table for the hash functions (every row will denote respectively: a,b,c coefficients for (a*x+b) % c hash-function
hash_table = get_hash_list(sig_size= signature_size, num_elem_univer_set=n)

#for loop should start here?
for row in range(n):
    for hash_func in range(signature_size):
        hash_value = hash_function(a=hash_table[hash_func][0], b = hash_table[hash_func][1], c= hash_table[hash_func][2], x= row)
        for col in range(m):
            if bin_matrix[row][col] == 1 and hash_value < minhash_matrix[hash_value][col]:
                minhash_matrix[hash_value][col] = hash_value














   