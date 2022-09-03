import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analyze(csv:str, arg_hist:str):
    '''
        LOADING THE CSV(S) SPLITTED AND ANALYZING IF THEY ARE EQUALLY DISTRIBUTED
        THE MAIN GOAL OF THIS FUNCTION IS TO SEE IF THE CLASSES HAVE
        (MORE OR LESS) THE SAME FREQUENCIES
    '''
    data = pd.read_csv(csv)
    len_df = len(data["cyberbullying_type"])
    counts = data["cyberbullying_type"].value_counts().to_dict()
    counts = {k: v/len_df for k,v in counts.items()}
    '''
        PRINTING SOME RESULTS
    '''
    print(2*'\n')
    print(50*'*')
    for k,v in counts.items():print(f'key:{k} // value: {v}')
    print(50*'*')
    print(2*'\n')
    
    fig = plt.figure()
    fig.set_figheight(5)
    fig.set_figwidth(10)
    plt.title(f'histogram of {arg_hist} set')
    # plt.scatter(np.asarray(list(counts.keys())),np.asarray(list(counts.values())))
    plt.pie(np.asarray(list(counts.values())),autopct= lambda x:round(x), labels = np.asarray(list(counts.keys())))
    plt.savefig(f'csv_data_info/histogram of {arg_hist} set.png', dpi = 450)
    plt.show()
    
def pie_charts_of_csvs():
    '''ANALYZE AND SAVE THE PIE CHARTS FOR ALL THE SETS'''
    train_path,validation_path,test_path = "./data/train_data.csv","./data/validation_data.csv","./data/test_data.csv"
    datasets = {"train":train_path,"validation":validation_path,"test":test_path}
    analyze("./data/cyberbullying_tweets.csv",'initial (unsplitted)')
    for set,path in datasets.items():analyze(path,set)
    
if __name__ == '__main__':
    pie_charts_of_csvs()