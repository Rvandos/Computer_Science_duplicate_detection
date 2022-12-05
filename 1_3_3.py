import matplotlib.pyplot as plt
import numpy as np


######################################
## Considering multiples of 3 and 5
######################################
#Get multiples of 3 and 5 up to 1000
multiple_3and5 = []
for i in range(0,1000,3):
    multiple_3and5.append(i)
for i in range(0,1000,5):
    if i % 3 != 0:
        multiple_3and5.append(i)


#Create a hash-key list with multiples of 3 and 5:
hash_key_multiple = []
for i in range(10000):
    rand_index = np.random.randint(0, len(multiple_3and5))
    hash_key_multiple.append(multiple_3and5[rand_index])

hash_result_multiple = []
for key in hash_key_multiple:
    hash_result_multiple.append(key%15)


plt.bar(hash_key_multiple,hash_result_multiple)
plt.show()


################################################
### Considering values not a multiple of 3 and 5
################################################

#draw values that are not a multiple of 3 or 5
k = 10000
hash_key_not_multiple = []
while k>0:
    a = int(np.random.randint(0,1000, size=1))
    while a % 3==0 or a%5==0:
            a = int(np.random.randint(2,1000, size=1))
    hash_key_not_multiple.append(a)
    k = k-1

hash_result_not_multiple = []
for key in hash_key_not_multiple:
    hash_result_not_multiple.append(key%15)


plt.bar(hash_key_not_multiple,hash_result_not_multiple)
plt.show()


