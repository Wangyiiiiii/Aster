"""
This script defines data reader
"""
import numpy as np
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def funcname(parameter_list):
    pass

def data_reader(data_name = "adult"):
    if(data_name == "adult"):
        #load data
        file_path = "./data/adult/"
        data1 = pd.read_csv(file_path + 'adult.data', header=None)
        data2 = pd.read_csv(file_path + 'adult.test', header=None)
        data2 = data2.replace(' <=50K.', ' <=50K')    
        data2 = data2.replace(' >50K.', ' >50K')
        
        data = pd.concat([data1,data2])
       
        #data transform: str->int
        data = np.array(data, dtype=str)
        labels = data[:,14]
        le= LabelEncoder()
        le.fit(labels)
        labels = le.transform(labels)
        data = data[:,:-1]
        
        categorical_features = [1,3,5,6,7,8,9,13]
        # categorical_names = {}
        for feature in categorical_features:
            le = LabelEncoder()
            le.fit(data[:, feature])
            data[:, feature] = le.transform(data[:, feature])
            # categorical_names[feature] = le.classes_
        data = data.astype(float)
        
        n_features = data.shape[1]
        numerical_features = list(set(range(n_features)).difference(set(categorical_features)))
        for feature in numerical_features:
            scaler = MinMaxScaler()
            sacled_data = scaler.fit_transform(data[:,feature].reshape(-1,1))
            data[:,feature] = sacled_data.reshape(-1)
        
        #OneHotLabel
        oh_encoder = ColumnTransformer(
            [('oh_enc', OneHotEncoder(sparse=False), categorical_features),], 
            remainder='passthrough' )
        oh_data = oh_encoder.fit_transform(data)
    

    elif(data_name == "bank"):
        #load data
        file_path = "./data/bank/"
        data = pd.read_csv(file_path + 'bank-full.csv',sep=';')
        #data transform
        data = np.array(data, dtype=str)
        labels = data[:,-1]
        le= LabelEncoder()
        le.fit(labels)
        labels = le.transform(labels)
        data = data[:,:-1]
        
        categorical_features = [1,2,3,4,6,7,8,10,15]
        # categorical_names = {}
        for feature in categorical_features:
            le = LabelEncoder()
            le.fit(data[:, feature])
            data[:, feature] = le.transform(data[:, feature])
            # categorical_names[feature] = le.classes_
        data = data.astype(float)
        
        n_features = data.shape[1]
        numerical_features = list(set(range(n_features)).difference(set(categorical_features)))
        for feature in numerical_features:
            scaler = MinMaxScaler()
            sacled_data = scaler.fit_transform(data[:,feature].reshape(-1,1))
            data[:,feature] = sacled_data.reshape(-1)
        #OneHotLabel
        oh_encoder = ColumnTransformer(
            [('oh_enc', OneHotEncoder(sparse=False), categorical_features),], 
            remainder='passthrough' )
        oh_data = oh_encoder.fit_transform(data)
        
    elif(data_name == "mnist"):
        file_path = "./data/mnist/"
        data = pd.read_csv(file_path + 'mnist_train.csv', header=None)
        data = np.array(data)
        labels = data[:,0]
        data = data[:,1:]
        
        categorical_features = []
        data = data/data.max()
        oh_encoder = ColumnTransformer(
            [('oh_enc', OneHotEncoder(sparse=False), categorical_features),], 
            remainder='passthrough' )
        oh_data = oh_encoder.fit_transform(data)
        
    else:
        
        str_list = data_name.split('_')
        file_path = "./data/purchase/"
        data = pd.read_csv(file_path+'dataset_purchase')
        data = np.array(data)
        data = data[:,1:]
        
        label_file = './data/purchase/label'+ str_list[1] + '.npy'
        
        labels = np.load(label_file)
        
        categorical_features = []
        oh_encoder = ColumnTransformer(
            [('oh_enc', OneHotEncoder(sparse=False), categorical_features),], 
            remainder='passthrough' )
        oh_data = oh_encoder.fit_transform(data)
        
        X_train, _, y_train, _ = train_test_split(oh_data, labels,test_size = 0.75)
        oh_data = X_train
        labels = y_train
        
    #randomly select 10000 records as training data
    train_idx = np.random.choice(len(labels), 10000, replace = False)
    idx = range(len(labels))
    idx = np.array(idx)
    test_idx = list(set(idx).difference(set(train_idx)))
    test_idx = np.array(test_idx)
    
    assert test_idx.sum() + train_idx.sum() == idx.sum()
    
    X_train = data[train_idx,:]
    Y_train = labels[train_idx]
    
    X_test = data[test_idx,:]
    Y_test = labels[test_idx]
    
    orig_dataset = {"X_train":X_train,
               "Y_train":Y_train,
               "X_test":X_test,
               "Y_test":Y_test}
    
    X_train = oh_data[train_idx,:]
    
    X_test = oh_data[test_idx,:]
    
    oh_dataset = {"X_train":X_train,
               "Y_train":Y_train,
               "X_test":X_test,
               "Y_test":Y_test}

    return orig_dataset, oh_dataset, oh_encoder
