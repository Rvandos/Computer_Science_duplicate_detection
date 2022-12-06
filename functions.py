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
