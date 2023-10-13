"""
This script defines some 
"""
from data_preprocessing import data_reader
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.svm import SVC 
from sklearn.linear_model import LogisticRegression as LR
from sklearn.neural_network import MLPClassifier as NN

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

    




