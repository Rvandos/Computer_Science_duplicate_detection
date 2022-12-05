#File with function to generate Hash-Functions:
import numpy as np
import sympy
###################################
# Hash function has the form: h(x) = (a*x + b)  % c,
# where a and b are random coefficients and c is chosen to be the next prime number of a pre-specified maximum
###################################


#Function to get list of unique a and b values for hash functions
def getCoefList(max_num, k):
    coeff_list = []
    while k>0:
        a = int(np.random.randint(2,max_num, size=1))
        while a in coeff_list:
            a = int(np.random.randint(2,max_num, size=1))
        coeff_list.append(a)
        k = k-1
    return coeff_list

#Function that outputs a list with a list of possible coefficients a,b and a fixed coefficient c
#number of elements in the universal set is multiplied by 5 to avoid collisison in same bucket.
#c is chosen to be the next largest prime. a and b cannot be 0 or 1, as this may lead to unevenly distributed hash keys
def hash_elements(sig_size, num_elem_univer_set):
	coefficients = []
	max_ab = num_elem_univer_set * 5
	c = sympy.nextprime(max_ab)
	ab_list = getCoefList(max_ab, sig_size*2)
	coefficients.append(ab_list)
	#print(nextprime(max_ab))
	coefficients.append(c)
	return(coefficients)
    
#Function to get the coefficient out of the coef_list and then removes the drawn numbers, to make sure no duplicates are used
def get_coef(hash_elements):
	ab_list = hash_elements[0]
	c = hash_elements[1]
	n = len(ab_list)

	#Get a coefficient
	a_index = np.random.randint(0, n)
	a = ab_list[a_index]
	ab_list.pop(a_index)

	#Get b coefficient
	b_index = np.random.randint(0,n-1)
	b = ab_list[b_index]
	ab_list.pop(b_index)

	hash_el_update = [ab_list, c]

	return([[a,b,c], hash_el_update])

def get_hash_list(sig_size, num_elem_univer_set):
	hash_table = np.zeros(shape=(sig_size,3))
	hash_el = hash_elements(sig_size, num_elem_univer_set)

	#Get first coefficient:
	for i in range(0,sig_size):
		update = get_coef(hash_el)
		coef = update[0]
		hash_el = update[1]

		#Update coefficients
		hash_table[i][0] = coef[0]
		hash_table[i][1] = coef[1]
		hash_table[i][2] = coef[2]
	return hash_table

#Literal hash-function
def hash_function(a,b,c,x):
	return((a*x + b) % c)
	

#Hash function sources:
#https://github.com/chrisjmccormick/MinHash/blob/master/runMinHashExample.py
#https://stackoverflow.com/questions/14533420/can-you-suggest-a-good-minhash-implementation  
#https://www.guru99.com/hash-table-data-structure.html

