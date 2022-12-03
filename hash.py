#File with function to generate Hash-Functions:
import numpy as np

###################################
# Hash function has the form: h(x) = (a*x + b)  % c,
# where a and b are random coefficients and c is chosen to be the next prime number of a pre-specified maximum
###################################

#Function to get the next largest prime
def nextprime(n):
	prime=0
	n+=1
	for i in range(2,int(n**0.5)+2):
		if n%i==0:
			prime=0
			break
		else:
			prime=1
	if prime==1:
		print(n)
		return
	else:
		nextprime(n)
		return


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


#Function that outputs a dictionary with a list of possible coefficients a,b and a fixed coefficient c
#number of elements in the universal set is multiplied by 53 to avoid collisison in same bucket.
#c is chosen to be the next largest prime. a and b cannot be 0 or 1, as this may lead to unevenly distributed hash keys
def hash_elements(sig_size, num_elem_univer_set):
    max_ab = num_elem_univer_set * 53
    c = nextprime(max_ab)
    coef_list = getCoefList(max_ab, sig_size*2)
    return {'ab' : coef_list, 'c' : c}
    

#Hash function sources:
#https://github.com/chrisjmccormick/MinHash/blob/master/runMinHashExample.py
#https://stackoverflow.com/questions/14533420/can-you-suggest-a-good-minhash-implementation  
#https://www.guru99.com/hash-table-data-structure.html

