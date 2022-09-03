import pandas as pd
import numpy as np 

def transform_key(key,dictionary_with_mapping):
    return dictionary_with_mapping[key[0]], dictionary_with_mapping[key[1]]

def retrieve_first_col(array):
    first_col = array[0]
    first_col = first_col[-1]
    first_col = int(first_col)
    return first_col
    
    
def retrieve_second_col(array):
    second_col = array[1]
    second_col = second_col[-2]
    second_col = int(second_col)
    return second_col

def analize_errors_of(path):
    df = pd.read_csv(path)
    labels_of_bullying = {0 : 'age',
                        1 : 'ethnicity',
                        2 : 'gender',
                        3 : 'not_cyberbullying',
                        4 : 'other_cyberbullying',
                        5 : 'religion'}
    array = df.to_numpy()
    first_column = []
    second_column = []
    # print(array)
    for el in array:
        first_column.append(retrieve_first_col(el))
        second_column.append(retrieve_second_col(el))
    first_column = np.asarray(first_column)
    second_column = np.asarray(second_column)
    # print(first_column)
    # print(second_column)
    
    counts = dict()
    
    for idx in range(len(first_column)):
        if first_column[idx]<second_column[idx]:
            couple = (first_column[idx],second_column[idx])
        else:
            couple = (second_column[idx],first_column[idx])

        if couple not in counts.keys():
            counts[couple] = 1
        else:
            counts[couple] += 1
    
    
    
    n_errors = len(counts)
    
    significant_errors_couples = [(3,0),(3,1),(3,2),(3,4),(3,5)]
    significant_errors = 0
    new_keys = []
    frequencies = list(counts.values())
    for k in counts.keys():
        new_key = transform_key(k,labels_of_bullying)
        new_keys.append(new_key)
    
    for k in counts.keys():
        if k in significant_errors_couples:
            significant_errors += 1
    
    # print(new_keys)
    
    
    counts = dict()
    for idx in range(len(frequencies)):
        counts[new_keys[idx]] = frequencies[idx]
    
    
            
    for k,v in counts.items():print(f'key: {k}  //  value: {v}')
    
    print(3*'\n',100*'=',3*'\n')
    
    print(f'the significant errors are {significant_errors}/{n_errors}')
    
    
    
if __name__ == "__main__":
    analize_errors_of('errors/Error_of_test.csv')
    print(3*'\n',50*'*',3*'\n')
    analize_errors_of('errors/Error_of_validation.csv')
    print(3*'\n',50*'*',3*'\n')
    analize_errors_of('errors/Supervised/Error_of_test.csv')
    print(3*'\n',50*'*',3*'\n')
    analize_errors_of('errors/Supervised/Error_of_validation.csv')
    
    