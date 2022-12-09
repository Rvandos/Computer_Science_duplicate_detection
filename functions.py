import numpy as np
import pandas as pd
import json
import regex as re
from hash import hash_function, get_hash_list
from sklearn.cluster import AgglomerativeClustering

#Function to pre-process the data 
def pre_process(data):
    df = []                                               
    mw_title = set()
    mw_value = set()
    for key in data.keys():
        for j in range(len(data[key])):
            #Add pre-processed data to new df
            df_el = {}  
            df_el[key] = []

            #Pre-process title:
            key_title = data[key][j]['title']

            #Products from Newegg website have the webiste url in front of title, remove those.
            key_title = key_title.replace('Newegg.com - ', "")

            #Remove refurbished in product titles
            key_title = key_title.replace('Refurbished: ',"")
            key_title = key_title.replace(' -', '').strip().lower()
            key_title = key_title.replace('refurbished', '')
            model_title = re.search("^[a-zA-Z0-9]+\s[0-9]+", key_title)
            if model_title != None:
                model_title = model_title.group().replace(" ", "")
                model_title = model_title.lower()
                df_el[key].append(model_title)
            #Add title of product to set object.
                mw_title.add(model_title)

            #Store website for non-duplicate detection:
            website = data[key][j]['shop']
            #Above does not do anything yet

            #pre-process key/value pairs:
            characteristics = data[key][j]['featuresMap']
            for items in characteristics.items():

                value = items[1]
                #Clean hertz inconsistencies:
                hertz_replace = ['Hertz', 'hertz', 'HZ','Hz', ' hz', '-hz']
                for char in hertz_replace:
                    value = value.replace(char, "hz")
                
                #Clean inch inconsistencies:
                inch_replace = ['Inch', 'inches', '"', '-inch', ' inch']
                for char in inch_replace:
                    value = value.replace(char, 'inch')

                #Clean Watt inconsistencies
                value = value.replace(' W', 'W')

                #Delete 'In', e.g. HDMI: '2 In'
                value = value.replace(' In', "")

                #Cast to lower case
                value = value.lower()

                value = re.search("[a-zA-Z0-9]*(([0-9]+[ˆ0-9, ]+)|([ˆ0-9, ]+[0-9]+))[a-zA-Z0-9]*", value)
                #Now if regex does not find a match it does not add the value of the key-value pair!!
                if value != None:
                    mw_value.add(value.group())
                    df_el[key].append(value.group())
            df.append(df_el)
    return(mw_title, mw_value, df)


#Function to obtain the binary matrix
def get_binary_matrix(mw_title, mw_value, processed_data):
    mw = set.union(mw_title, mw_value)
    n = len(mw)
    m = len(processed_data)
    bin_matrix = np.zeros((n,m))
    for i in range(m):
        element = processed_data[i]
        for key in element.keys():                       #There will only be one key every time!
            k=0
            for model_word in mw:
                for item in element[key]:
                    if(model_word == item):
                        bin_matrix[k, i] = 1
                k = k+1
    return bin_matrix


#Function that performs minhashing and outputs the signature matrix. 
#Input sig_ratio is the percentage of elements in the universal sets to keep (number between 0 and 1)
def get_signature_matrix(binary_matrix, signature_size):
    n = len(binary_matrix)
    m = len(binary_matrix[0]) 

    #For the products which do not match any title or feature, add a random number so to not get an error, temporary fix!?
    troublesome_index = []
    for i in range(m):
        count = 0
        for j in range(n):
            if binary_matrix[j][i] == 1:
                count = count +1
        if count == 0:
            troublesome_index.append(i)
    for i in troublesome_index:
        fix = np.random.randint(n)
        binary_matrix[fix][i] = 1

    minhash_matrix = np.ones(shape=(signature_size, m)) * np.inf

    #Get hash table for the hash functions (every row will denote respectively: a,b,c coefficients for (a*x+b) % c hash-function
    hash_table = get_hash_list(sig_size= signature_size, num_elem_univer_set=n*100)

    for row in range(n):
        print('Progress: ',row, ' of ', n)
        for hash_func in range(signature_size):
            hash_value = int(hash_function(a=hash_table[hash_func][0], b = hash_table[hash_func][1], c= hash_table[hash_func][2], x= row))
            for col in range(m):
                if binary_matrix[row][col] == 1 and hash_value < minhash_matrix[hash_func][col]:
                    minhash_matrix[hash_func][col] = hash_value
    return minhash_matrix


#Outputs a list with possible candidate tuples (b,r) for (#bands, #rows) for which b * r = n
def get_bandrow(sig_matrix, min_bands, max_bands):
    n = len(sig_matrix)
    bandrow_list = []
    for bands in range(min_bands, max_bands):
        if n % bands == 0:
            r = n/bands
            bandrow_list.append((bands,r))
    return bandrow_list


#LSH function returns a set of tuples involving the products that are candidates.
#Out of all the possible band choices (s.t. b * r = n), at least 40 bands have to be chosen, otherwise it crashes as concatenating 
#40 numbers will be too large to convert back to an integer
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
        #the number of buckets used).
        hash_table = get_hash_list(sig_size= bands, num_elem_univer_set= n*100000)
       
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


#Idea for clustering: perhaps take the number of times products are hashed to each other into account, now considers only cases >=1
#Not efficient as it takes the full matrix instead of just the upper (bottom) half:
def get_distance_matrix(candidate_pairs, signature_matrix):
    m = len(signature_matrix[0])
    distance_matrix = np.ones(shape=(m, m)) * 1000000000
    diagonal = np.identity(m) * 1000000000
    distance_matrix = distance_matrix - diagonal

    for candidate in candidate_pairs:
        cand_1 = candidate[0]
        cand_2 = candidate[1]
        #Extract columns of products in signature matrix:
        sig_1 = signature_matrix[:,cand_1]
        sig_2 = signature_matrix[:,cand_2]
        #Get similarity between sig_1 and sig_2
        similarity_value = cosine_similarity(sig_1,sig_2)
        inverse_sim = 1/similarity_value
        #update (inversed similarity) in distance matrix (note the symmetry):
        distance_matrix[cand_1,cand_2] = inverse_sim
        distance_matrix[cand_2,cand_1] = inverse_sim
    return distance_matrix


#Function to do the clustering, returns a list of duplicate tuple-sets
def clustering(distance_matrix, dist_threshold):
    clustering_object = AgglomerativeClustering(affinity='precomputed', linkage= 'single', distance_threshold= dist_threshold, n_clusters=None).fit(distance_matrix)
    cluster_assignment = clustering_object.labels_
    #Get a list with the found duplicate pairs.
    n = len(cluster_assignment)
    cluster_pairs = set()
    for i in range(n-1):
        for j in range(i+1, n):
            if cluster_assignment[i] == cluster_assignment[j]:
                cluster_pairs.add((i,j))
    return cluster_pairs


#Function that can handle the fact that keys are unique in dictionaries!
def add_to_dict(dict, key, value):
    #Case 1). The key is not yet added to the dict
    if key not in dict.keys(): 
        dict[key] = value
    #Case 2). The key is contained more than twice (value already converted to list)  
    elif type(dict[key]) == list:
        dict[key].append(value)
    #Case 3). The key is contained for the second time, replace value and put it in a list.
    else:
        #Put old value in list
        old_value = dict[key]
        new_list = [old_value]
        dict[key] = new_list
        #append new value
        dict[key].append(value)
    return(dict)


#Function that only adds tuple (x,y) to a set when both (x,y) and (y,x) are not yet contained in the set
def add_to_set(set_obj, tuple_pair):
    x = tuple_pair[0]
    y = tuple_pair[1]
    if tuple_pair not in set_obj and (y,x) not in set_obj:
        set_obj.add(tuple_pair)
    return set_obj


#Function to get the cosine_similarity
def cosine_similarity(sig_1, sig_2):
    numerator = np.dot(np.transpose(sig_1),sig_2)
    denominator = np.linalg.norm(sig_1) * np.linalg.norm(sig_2)
    return (numerator/denominator)


#Function to get the smallest similarity (largest distance) between real duplicates
#This can then be used on the train data to set a threshold for the clusters to form.
def get_cluster_threshold(train_data, signature_matrix_train):
    #First get the real duplicate pairs:
    dup_pairs = get_true_dup_pairs(train_data)

    min_similarity = np.inf
    for pair in dup_pairs:
        pair = list(pair)
        el1 = pair[0]
        el2 = pair[1]
        similarity = cosine_similarity(signature_matrix_train[:,el1], signature_matrix_train[:,el2])
        if similarity < min_similarity:
            min_similarity = similarity
    max_distance = 1/ min_similarity
    return max_distance


#Function that splits the data into train-test, based on bootstrapping the number of products (with replacement)
#Train data is the bootstrapped data, test data is the remaining part of the data
def boot_strap_split(processed_data):
    n = len(processed_data)

    draw_list = set()
    remaining = set([i for i in range(n)])
    for i in range(n):
        draw = np.random.randint(0, n)
        draw_list.add(draw)
        try:
            remaining.remove(draw)
        except:
            pass

    #Append train data
    draw_list = list(draw_list)
    train_data = []
    for i in range(len(draw_list)):
        train_data.append(processed_data[draw_list[i]])
    #Append remaining test data
    test_data = []
    for j in range(len(remaining)):
        test_data.append(processed_data[list(remaining)[j]])
    return train_data, test_data
    

#Function to get the true pairs of duplicates, takes as input a list with dictionaries representing products
#Outputs a list with sets, where indices are used to refer to products.
#for every product with n duplicates --> n(n-1)/2 pair representations.
def get_true_dup_pairs(data):
    dup_dict = {}
    dup_list = set()
    n = len(data)
    for i in range(n):
        for model_id in data[i].keys():              #Everytime contains only one key!
            #Here the dictionary will have as keys the model_ids and the values will denote the corresponding index
            #if the value is of type list, there will be duplicates.
            add_to_dict(dup_dict, model_id, i)
    for key in dup_dict.keys():
        temp = dup_dict[key]
        if type(temp)==list:
            for i in range(len(temp)-1):
                for j in range(i+1,len(temp)):
                    dup_pair = (temp[i],temp[j])
                    dup_list.add(dup_pair)
    return dup_list


#Function to get the f1 performance score, input: sets with tuples of candidates (already in percentages)
def get_performance(found_dup, true_dup):
    #Find the sets that are in both lists 
    tp = len(set.intersection(found_dup,true_dup))
    try:
        precision = tp / len(found_dup)
    except:
        precision = 0 
    try:
        recall = tp / len(true_dup)
    except:
        recall = 0
    try:
        f1_score = 2 * (precision * recall) / (precision + recall)
    except:
        f1_score = 0
    performance_dic = {'tp': tp , 'precision': precision, 'recall': recall, 'f1':f1_score}
    return performance_dic


#Function to get the pair quality, pair completeness, etc.
def get_scalability_performance(candidate_pairs, found_dup, true_dup):
    dup_correct = len(set.intersection(found_dup, true_dup))
    comparisons_lsh = len(candidate_pairs)
    total_dup = len(true_dup)

    pair_quality = dup_correct / comparisons_lsh
    pair_completeness = dup_correct / total_dup
    try:
        f1_score = 2 * (pair_quality * pair_completeness) / (pair_quality + pair_completeness)
    except:
        f1_score = 0

    performance_dic = {'pair quality': pair_quality, 'pair completeness': pair_completeness, 'f1': f1_score}
    return performance_dic