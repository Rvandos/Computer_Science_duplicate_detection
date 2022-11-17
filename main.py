import json
import pandas as pd
import numpy as np
import regex as re

with open('TVs-all-merged.json') as f:
    data = json.load(f)

    
#get all elements                          
df = {}
mw = set()
for key in data.keys():
    #Pre-process title:
    key_title = data[key][0]['title']
    model_word = re.search("[a-zA-Z0-9]*(([0-9]+[ˆ0-9, ]+)|([ˆ0-9, ]+[0-9]+))[a-zA-Z0-9]*", key_title)
    
    #Add title of product to set object.
    mw.add(model_word.group().replace(" ", ""))

    #pre-process key/value pairs:
    characteristics = data[key][0]['featuresMap']

    for items in characteristics.items():
        
        value = re.search("[a-zA-Z0-9]*(([0-9]+[ˆ0-9, ]+)|([ˆ0-9, ]+[0-9]+))[a-zA-Z0-9]*", items[1])
        if value != None:
            mw.add(value.group())



#Binary matrix:
n = len(mw)
m = len(data.keys())
bin_matrix = np.zeros((n,m))



   