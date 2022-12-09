import pandas as pd
import numpy as np
import json
import functions

#Dictionary to store lsh performance, keys denote the bands used, every key will have a corresponding list of the form:
# [fraction of comparison, pair quality, pair completeness, f1*]
lsh_performance_dic = {}

#Dictionary to store overall performance, keys denote the bands used, every key will have a corresponding list of the form:
# [fraction of comparisons, precision, recall, f1]
overall_performance_dic = {}


np.random.seed(6)

#For all bootstraps get performance measurements!
for bootstrap in range(5):
    print("processing bootsrap: ", bootstrap)
    ##########################################################################
    ### First load in the signature matrix and the train, test data sets
    ##########################################################################

    #Signature matrices
    train_sig_matrix = pd.read_csv('bootstrap_input/Signature_matrix_train_bootstrap' + str(bootstrap) + '.csv')
    train_sig_matrix = train_sig_matrix.drop('Unnamed: 0', axis=1).to_numpy()

    test_sig_matrix = pd.read_csv('bootstrap_input/Signature_matrix_test_bootstrap' + str(bootstrap) + '.csv')
    test_sig_matrix = test_sig_matrix.drop('Unnamed: 0', axis=1).to_numpy()

    #Import train and test data
    with open('bootstrap_input/train_data_bootstrap' + str(bootstrap) + '.json', 'r') as f:
        train_data = json.load(f)
    f.close()

    with open('bootstrap_input/test_data_bootstrap' + str(bootstrap) + '.json', 'r') as f:
        test_data = json.load(f)
    f.close()

    #Get candidates for the different number of bands:
    band_row = functions.get_bandrow(train_sig_matrix, 40, 1600)
    band_list = []
    for element in band_row:
        band_list.append(element[0])


    #Perform analysis for multiple number of bands
    for band in band_list:
        print('Processing band: ', band)
        ##########################################################################
        ### Training part of the algorithm:
        ##########################################################################

        #Perform LSH on train-set   
        candidate_pairs_train = functions.LSH(train_sig_matrix, band)

        #Get the input for the clustering (distance matrix and cluster threshold)
        dist_matrix_train = functions.get_distance_matrix(candidate_pairs_train, train_sig_matrix)
        cluster_threshold = functions.get_cluster_threshold(train_data, train_sig_matrix) 



        ##########################################################################
        ### Testing part of the algorithm:
        ##########################################################################

        #Perform LSH on test-set
        candidate_pairs_test = functions.LSH(test_sig_matrix, band)

        #Get the distance matrix and cluster results
        dist_matrix_test = functions.get_distance_matrix(candidate_pairs_train, train_sig_matrix)
        found_duplicates_test = functions.clustering(dist_matrix_test, cluster_threshold)

        #Get performance results
        true_duplicates_test = functions.get_true_dup_pairs(test_data)

        #LSH/scalability performance
        lsh_performance = functions.get_scalability_performance(candidate_pairs_test, found_duplicates_test, true_duplicates_test)
        overall_performance = functions.get_performance(found_duplicates_test, true_duplicates_test)



        ##########################################################################
        ### Store performance measurements
        ##########################################################################

        #Store results for every band
        total_number_comparisons = len(test_data) * (len(test_data)-1)/2
        fraction_comparisons = len(candidate_pairs_test) / total_number_comparisons

        #Store scalability results (LSH)
        to_add = np.array([fraction_comparisons, lsh_performance['pair quality'], lsh_performance['pair completeness'], lsh_performance['f1']])
        if band in lsh_performance_dic.keys():
            lsh_performance_dic[band] = lsh_performance_dic[band] + to_add
        else:
            lsh_performance_dic[band] = to_add

        #Store overall performance results
        to_add2 = np.array([fraction_comparisons, overall_performance['precision'], overall_performance['recall'], overall_performance['f1']])
        if band in overall_performance_dic.keys():
            overall_performance_dic[band] = overall_performance_dic[band] + to_add2
        else:
            overall_performance_dic[band] = to_add2



#average across performance, so divide by 5
for key in lsh_performance_dic:
    lsh_performance_dic[key] = list(lsh_performance_dic[key] /5)
    overall_performance_dic[key] = list(overall_performance_dic[key] /5)


#Write the results to seperate files:
with open('LSH_performance.json', 'w') as f:
    f.write(json.dumps(lsh_performance_dic))
f.close()

with open('overall_performance.json', 'w') as f:
    f.write(json.dumps(overall_performance_dic))
f.close()


