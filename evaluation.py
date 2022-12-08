import pandas as pd
import numpy as np
import json
import functions



###########################################################################
### First get the Data and pre-process:
###########################################################################
with open('TVs-all-merged.json') as f:
    data = json.load(f)
mw_title, mw_value, processed_data = functions.pre_process(data)



# 1). Get 5 bootstraps: the splitted train and test data, and signature matrices are stored in seperate files 
# as they take a long time to load.
np.random.seed(0) 
for bootstrap in range(5):
    ###########################################################################
    ### Split the data into train and test part based on bootstrapping:
    ###########################################################################
    train_data, test_data = functions.boot_strap_split(processed_data)

    #Store the train and test data files seperately
    train_data_json = json.dumps(train_data)
    test_data_json = json.dumps(test_data)
    with open('train_data_bootstrap'+ str(bootstrap)+'.json', 'w') as f:
        f.write(train_data_json)
    f.close()

    with open('test_data_bootstrap'+ str(bootstrap)+'.json', 'w') as f:
        f.write(test_data_json)
    f.close()

    ###########################################################################
    ### Compute the signature matrix for train data:
    ###########################################################################
    train_bin_matrix = functions.get_binary_matrix(mw_title,mw_value,train_data)
    train_sig_matrix = functions.get_signature_matrix(train_bin_matrix, 1600)
    #Save signature matrix into csv file
    pd.DataFrame(train_sig_matrix).to_csv('Signature_matrix_train_bootstrap' + str(bootstrap) + '.csv')


    ###########################################################################
    ### Compute the signature matrix for test data:
    ###########################################################################
    test_bin_matrix = functions.get_binary_matrix(mw_title,mw_value, test_data)
    test_sig_matrix = functions.get_signature_matrix(test_bin_matrix, 1600)
    #Save signature matrix into csv file
    pd.DataFrame(test_sig_matrix).to_csv('Signature_matrix_test_bootstrap' + str(bootstrap) + '.csv')



"""
#Use code below when loading a signature matrix csv file
train_sig_matrix = pd.read_csv('Signature_matrix_train.csv')
train_sig_matrix = train_sig_matrix.drop('Unnamed: 0', axis=1).to_numpy()

#Get cluster threshold
cluster_threshold = functions.get_cluster_threshold(train_data, train_sig_matrix)
print(cluster_threshold)

#LSH part



###########################################################################
### Test data 
###########################################################################
pd.DataFrame(test_sig_matrix).to_csv('Signature_matrix_test.csv')
test_sig_matrix = pd.read_csv('Signature_matrix_test.csv')
test_sig_matrix = test_sig_matrix.drop('Unnamed: 0', axis=1).to_numpy()



#LSH part:
candidate_pairs_test = functions.LSH(test_sig_matrix,50)
dist_matrix_test = functions.get_distance_matrix(candidate_pairs_test, test_sig_matrix)



#Obtain performance
found_duplicates_test = functions.clustering(dist_matrix_test, cluster_threshold) 
true_duplicates_test = functions.get_true_dup_pairs(test_data)

performance_cluster = functions.get_performance(found_duplicates_test, true_duplicates_test)
performance_lsh = functions.get_scalability_performance(candidate_pairs_test, found_duplicates_test, true_duplicates_test)
#print(performance_lsh)

"""