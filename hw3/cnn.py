import os, sys
import numpy as np
import pandas as pd


def load_data(train_data_path, test_data_path):
    df_train = pd.read_csv(train_data_path)
    df_test = pd.read_csv(test_data_path)
    
    #train_data
    X_train = []
    for i in range(df_train['feature'].shape[0]):
	X_train.append(list(map(int,df_train['feature'][i].split())))
    X_train = np.array(X_train)
    y_train = df_train['label'].values

    #test_data
    X_test = []
    for i in range(df_test['feature'].shape[0]):
        X_test.append(list(map(int,df_test['feature'][i].split())))
    X_test = np.array(X_test)

    return (X_train, y_train, X_test)

