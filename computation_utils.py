"""
This script defines some computation functions and other utilities
"""
import random

import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.svm import SVC 
from sklearn.linear_model import LogisticRegression as LR
from sklearn.neural_network import MLPClassifier as NN

from data_preprocessing import data_reader

categorical_list ={
    "adult": [1,3,5,6,7,8,9,13],
    "bank": [1,2,3,4,6,7,8,10,15],
    "null": [],
}

def fn_Sample_Generator(R_given, dataset):
    if not dataset in categorical_list.keys():
        dataset = "null"
    epsilon = 1e-6
    R_given = R_given.reshape([1, -1])
    n_feature = R_given.shape[1]
    local_samples = np.repeat(R_given, repeats=n_feature, axis=0)
    for i in range(n_feature):
        if i in categorical_list[dataset]:
            continue
        local_samples[i][i] += epsilon

    return local_samples

def fn_Jacobian_Calculation(R_given, local_proba, n_features, n_class):
    epsilon = 1e-6
    jacobian = np.zeros([n_class, n_features])

    for ii in range(n_class):
        jacobian[ii, :] = (local_proba[:, ii] - R_given[ii]) / epsilon
    return jacobian

def fn_random_perturb(R_given, n_sample, categorical, feature_dict, max_amplitude=0.5,):
    n_feature = R_given.shape[0]
    perturbed = np.zeros((n_sample, n_feature))
    perturbed[0] = R_given
    for i in range(n_sample-1):
        perturb_sample = R_given
        feature_to_perturb = random.sample(range(n_feature), n_feature//2)
        for feature_index in feature_to_perturb:
            if feature_index in categorical:
                perturb_sample[feature_index] = random.choice(feature_dict[feature_index])
            else:
                perturb_sample[feature_index] *= (1 + random.uniform(-max_amplitude, max_amplitude))
        perturbed[i+1] = perturb_sample
    return perturbed

def Target_Model_pred_fn(Target_Model, X_test):
    if(isinstance(Target_Model, RF)):
        pred_proba = Target_Model.predict_proba(X_test)
    elif(isinstance(Target_Model, LR)):
        pred_proba = Target_Model.predict_proba(X_test)
    else:
        pred_proba = Target_Model.predict_proba(X_test)
    return pred_proba

def fn_R_given_Selected(dataset, IN_or_OUT = 1):
    if(IN_or_OUT == 1):#IN_or_OUT == 1 meaning selecting R_given from training set
        idx = np.random.choice( len(dataset['Y_train']) )
        R_given = dataset['X_train'][idx,:]
        R_given_y = dataset['Y_train'][idx]
    elif(IN_or_OUT == 0):#IN_or_OUT == 0 meaning selecting R_given from testing set
        idx = np.random.choice( len(dataset['Y_test']) )
        R_given = dataset['X_test'][idx,:]
        R_given_y = dataset['Y_test'][idx]
    return R_given, R_given_y

def Target_Model_pred_fn(Target_Model, X_test):
    if(isinstance(Target_Model, RF)):
        pred_proba = Target_Model.predict_proba(X_test)
    elif(isinstance(Target_Model, LR)):
        pred_proba = Target_Model.predict_proba(X_test)
    else:
        pred_proba = Target_Model.predict_proba(X_test)
    return pred_proba

def fn_R_given_Selected(dataset, IN_or_OUT = 1):
    if(IN_or_OUT == 1):#IN_or_OUT == 1 meaning selecting R_given from training set
        idx = np.random.choice( len(dataset['Y_train']) )
        R_given = dataset['X_train'][idx,:]
        R_given_y = dataset['Y_train'][idx]
    elif(IN_or_OUT == 0):#IN_or_OUT == 0 meaning selecting R_given from testing set
        idx = np.random.choice( len(dataset['Y_test']) )
        R_given = dataset['X_test'][idx,:]
        R_given_y = dataset['Y_test'][idx]
    return R_given, R_given_y