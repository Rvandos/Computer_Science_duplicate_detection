import pandas as pd 
import numpy as np
import json
import matplotlib.pyplot as plt


#####################################################################################################
### LSH performance plot (efficiency)
#####################################################################################################

#First load the results data:
# Keys refer to number of bands --> value list: [fraction of comparison, pair quality, pair completeness, f1*]
with open('results/LSH_performance.json', 'r') as f:
    lsh_results = json.load(f)
f.close()

#Get necessary data for plots
frct_comp = []
pq = []
pc = []
f1_star = []

for key in lsh_results.keys():
    frct_comp.append(lsh_results[key][0])
    pq.append(lsh_results[key][1])
    pc.append(lsh_results[key][2])
    f1_star.append(lsh_results[key][3])

#Pair quality plot:
plt.plot(frct_comp, pq)
plt.xlabel('Fraction of comparisons')
plt.ylabel('Pair quality')
plt.savefig('figures/pq.png')
plt.show()

#Pair completeness plot:
plt.plot(frct_comp, pc)
plt.xlabel('Fraction of comparisons')
plt.ylabel('Pair completeness')
plt.savefig('figures/pc.png')
plt.show()

#F1* plot:
plt.plot(frct_comp, f1_star)
plt.xlabel('Fraction of comparisons')
plt.ylabel('F1*')
plt.savefig('figures/f1_star.png')
plt.show()




#####################################################################################################
### LSH performance plot (efficiency)
#####################################################################################################

#First load the results data:
# Keys refer to number of bands --> value list: [fraction of comparisons, precision, recall, f1]
with open('results/overall_performance.json', 'r') as f:
    overall_performance = json.load(f)
f.close()

#Get necessary data:
precision = []
recall = []
f1 = []

for key in overall_performance.keys():
    precision.append(overall_performance[key][1])
    recall.append(overall_performance[key][2])
    f1.append(overall_performance[key][3])

#Precision plot:
plt.plot(frct_comp, precision)
plt.xlabel('Fraction of comparisons')
plt.ylabel('Precision')
plt.savefig('figures/precision.png')
plt.show()

#Recall plot:
plt.plot(frct_comp, recall)
plt.xlabel('Fraction of comparisons')
plt.ylabel('Recall')
plt.savefig('figures/recall.png')
plt.show()

#F1 plot:
plt.plot(frct_comp, f1)
plt.xlabel('Fraction of comparisons')
plt.title('F1-score')
plt.savefig('figures/f1.png')
plt.show()
