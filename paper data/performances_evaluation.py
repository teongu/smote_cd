import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, f1_score, mean_squared_error
from scipy.stats import gmean, kde
import scipy.stats as st
from sklearn.model_selection import KFold
import smote_cd

from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense

from multiprocessing import Pool

import rpy2
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

# Defining the R script and loading the instance in Python
robjects.r('''library (DirichletReg)
library(MLmetrics)''')
robjects.r('''library (DirichletReg)
library(MLmetrics)''')

r = robjects.r
r['source']('dirichletReg.R')
# Loading the function we have defined in R.
dirichlet_function_r = robjects.globalenv['dirichlet_model']



##################################################
##### METRICS 
##################################################

def dirichlet_model(X_train, y_train, X_test, y_test, n_features):
    input_train=pd.DataFrame(np.concatenate((X_train,y_train),axis=1))
    input_test=pd.DataFrame(np.concatenate((X_test,y_test),axis=1))
    with localconverter(robjects.default_converter + pandas2ri.converter):
        input_train_r = robjects.conversion.py2rpy(input_train)
        input_test_r = robjects.conversion.py2rpy(input_test)
    df_result_r = dirichlet_function_r(input_train_r, input_test_r, n_features)
    return(np.array(df_result_r[0]),np.array(df_result_r[1]))

def crossentropy(y_true, y_pred):
    eps=1e-200
    return np.nanmean(-(y_true * np.log(y_pred+eps)).sum(axis=1))

def aic(y_true,y_pred,X):
    sse=sum((y_true-y_pred)**2)
    return(2*len(X)-2*np.log(sse))

def accuracy(y_true,y_pred):
    return sum(np.argmax(y_true,axis=1)==np.argmax(y_pred,axis=1))/len(y_true)

def logratio_transform(p0):
    if len(np.shape(p0))==1:
        p=np.copy(p0)
        p[p==0]=1e-20
        lrp=np.log(p/gmean(p))
    else:
        lrp=np.array([logratio_transform(pi) for pi in p0])
    return(lrp)

def softmax(z):
    if len(np.shape(z))==1:
        result=np.exp(z)/np.sum(np.exp(z))
        result[result<=1e-19]=0
    else:
        result=np.array([softmax(zi) for zi in z])
    return(result)


##################################################
##### p-value
##################################################

def pvalue(mean1, mean2, std1, std2, n):
    s1 = std1/np.sqrt(n-1)
    s2 = std2/np.sqrt(n-1)
    tt = (mean1-mean2)/np.sqrt(s1**2 + s2**2)
    return(st.t.sf(np.abs(tt), 2*n-2)*2)


##################################################
##### GB FUNCTIONS
##################################################

def gb_predict(X_train, y_train, X_test):
    y_pred_gb=[]
    for i_class in range(np.shape(y_train)[1]):
        reg = GradientBoostingRegressor(random_state=2)
        reg.fit(X_train, y_train[:,i_class])
        y_pred_gb.append(reg.predict(X_test))
    y_pred_gb=np.transpose(y_pred_gb)
    return(y_pred_gb)

def gb_predict_raw(X_train, y_train, X_test):
    y_pred_gb=[]
    for i_class in range(np.shape(y_train)[1]):
        reg = GradientBoostingRegressor(ccp_alpha = 10, learning_rate = 0.01, max_depth = 5, max_features = 'log2',
                               min_samples_leaf = 10, n_estimators = 200, random_state=2)
        reg.fit(X_train, y_train[:,i_class])
        y_pred_gb.append(reg.predict(X_test))
    y_pred_gb=np.transpose(y_pred_gb)
    return(y_pred_gb)

def gb_predict_compositional(X_train, y_train, X_test):
    y_pred_gb=[]
    for i_class in range(np.shape(y_train)[1]):
        reg = GradientBoostingRegressor(ccp_alpha = 10, learning_rate = 0.01, max_depth = 5, max_features = 'sqrt',
                               min_samples_leaf = 1, n_estimators = 100, random_state=2)
        reg.fit(X_train, y_train[:,i_class])
        y_pred_gb.append(reg.predict(X_test))
    y_pred_gb=np.transpose(y_pred_gb)
    return(y_pred_gb)

def gb_predict_logratio(X_train, y_train, X_test):
    y_pred_gb=[]
    for i_class in range(np.shape(y_train)[1]):
        reg = GradientBoostingRegressor(ccp_alpha = 0.5, learning_rate = 0.01, max_depth = 4, max_features = 'sqrt',
                               min_samples_leaf = 10, n_estimators = 200, random_state=2)
        reg.fit(X_train, y_train[:,i_class])
        y_pred_gb.append(reg.predict(X_test))
    y_pred_gb=np.transpose(y_pred_gb)
    return(y_pred_gb)



##################################################
##### PERFORMANCES EVALUATION FUNCTIONS
##################################################

n_features=16
n_classes=4
size_sample=550
betas_init=smote_cd.dataset_generation.generate_betas(n_features,n_classes,random_state=4)
betas_init[0,0]=1

def perf_evaluation_dirichlet_repetition(j, n_features=n_features, n_classes=n_classes, size_sample=size_sample, 
                                         betas_init=betas_init, n_iter=100, verbose=False, step=1, undersampling=False):
    
    betas=np.copy(betas_init)
    betas[0,0]+=j
        
    r2_compositional=[]
    r2_raw=[]
    r2_logratio=[]
    accuracy_compositional=[]
    accuracy_raw=[]
    accuracy_logratio=[]
    f1_compositional=[]
    f1_raw=[]
    f1_logratio=[]

    imbalance_ratio=[]

    for i in range(n_iter):

        X,y,_=smote_cd.dataset_generation.generate_dataset(n_features,n_classes,size_sample,betas=betas,random_state=i)
        imbalance_ratio.append(sum(y)[0]/np.sum(y))

        if undersampling:
            undersampled_indexes = smote_cd.random_undersampling(y)
            indexes_to_keep = [i for i in range(len(y)) if i not in undersampled_indexes]
            y_for_os = y[indexes_to_keep]
            X_for_os = X[indexes_to_keep]
        else:
            y_for_os = np.copy(y)
            X_for_os = np.copy(X)

        X_test,y_test,_=smote_cd.dataset_generation.generate_dataset(n_features,n_classes,size_sample*20,betas=betas,random_state=i+9999)

        X_logratio,y_logratio=smote_cd.oversampling_multioutput(X_for_os,y_for_os,label_distance='logratio',n_iter_max=2e3,k=10,norm=2)
        X_compositional,y_compositional=smote_cd.oversampling_multioutput(X_for_os,y_for_os,label_distance='compositional',n_iter_max=2e3,k=10,norm=2)


        _,y_compositional_predict=dirichlet_model(X_compositional, y_compositional, X_test, y_test, n_features)
        _,y_predict=dirichlet_model(X, y, X_test, y_test, n_features)
        _,y_logratio_predict=dirichlet_model(X_logratio, y_logratio, X_test, y_test, n_features)

        r2_raw.append(r2_score(y_test, y_predict, multioutput='raw_values'))
        r2_compositional.append(r2_score(y_test, y_compositional_predict, multioutput='raw_values'))
        r2_logratio.append(r2_score(y_test, y_logratio_predict, multioutput='raw_values'))

        accuracy_raw.append(accuracy(y_test, y_predict))
        accuracy_compositional.append(accuracy(y_test, y_compositional_predict))
        accuracy_logratio.append(accuracy(y_test, y_logratio_predict))

        f1_raw.append(f1_score(np.argmax(y_test,axis=1),np.argmax(y_predict,axis=1),average=None))
        f1_compositional.append(f1_score(np.argmax(y_test,axis=1),np.argmax(y_compositional_predict,axis=1),average=None))
        f1_logratio.append(f1_score(np.argmax(y_test,axis=1),np.argmax(y_logratio_predict,axis=1),average=None))
        
    r2_compositional_tot = np.round(np.mean(r2_compositional,axis=0),5)
    r2_raw_tot = np.round(np.mean(r2_raw,axis=0),5)
    r2_logratio_tot = np.round(np.mean(r2_logratio,axis=0),5)
    accuracy_compositional_tot = np.round(np.mean(accuracy_compositional),5)
    accuracy_raw_tot = np.round(np.mean(accuracy_raw),5)
    accuracy_logratio_tot = np.round(np.mean(accuracy_logratio),5)
    f1_compositional_tot = np.round(np.mean(f1_compositional,axis=0),5)
    f1_raw_tot = np.round(np.mean(f1_raw,axis=0),5)
    f1_logratio_tot = np.round(np.mean(f1_logratio,axis=0),5)
    imbalance_ratio_tot = np.round(np.mean(imbalance_ratio),5)

    r2_compositional_std = np.round(np.std(r2_compositional,axis=0),5)
    r2_raw_std = np.round(np.std(r2_raw,axis=0),5)
    r2_logratio_std = np.round(np.std(r2_logratio,axis=0),5)
    accuracy_compositional_std = np.round(np.std(accuracy_compositional),5)
    accuracy_raw_std = np.round(np.std(accuracy_raw),5)
    accuracy_logratio_std = np.round(np.std(accuracy_logratio),5)
    f1_compositional_std = np.round(np.std(f1_compositional,axis=0),5)
    f1_raw_std = np.round(np.std(f1_raw,axis=0),5)
    f1_logratio_std = np.round(np.std(f1_logratio,axis=0),5)
    imbalance_ratio_std = np.round(np.std(imbalance_ratio),5)
        
    return(r2_compositional_tot, r2_raw_tot, r2_logratio_tot,
           accuracy_compositional_tot, accuracy_raw_tot, accuracy_logratio_tot,
           f1_compositional_tot, f1_raw_tot, f1_logratio_tot, 
           imbalance_ratio_tot,
           r2_compositional_std, r2_raw_std, r2_logratio_std,
           accuracy_compositional_std, accuracy_raw_std, accuracy_logratio_std,
           f1_compositional_std, f1_raw_std, f1_logratio_std, 
           imbalance_ratio_std)



def perf_evaluation_gb_repetition(j, n_features=n_features, n_classes=n_classes, size_sample=size_sample, 
                                         betas_init=betas_init, n_iter=100, verbose=False, step=1, undersampling=False):
    
    betas=np.copy(betas_init)
    betas[0,0]+=j
        
    r2_compositional=[]
    r2_raw=[]
    r2_logratio=[]
    accuracy_compositional=[]
    accuracy_raw=[]
    accuracy_logratio=[]
    f1_compositional=[]
    f1_raw=[]
    f1_logratio=[]

    imbalance_ratio=[]

    for i in range(n_iter):

        X,y,_=smote_cd.dataset_generation.generate_dataset(n_features,n_classes,size_sample,betas=betas,random_state=i)
        imbalance_ratio.append(sum(y)[0]/np.sum(y))

        if undersampling:
            undersampled_indexes = smote_cd.random_undersampling(y)
            indexes_to_keep = [i for i in range(len(y)) if i not in undersampled_indexes]
            y_for_os = y[indexes_to_keep]
            X_for_os = X[indexes_to_keep]
        else:
            y_for_os = np.copy(y)
            X_for_os = np.copy(X)

        X_test,y_test,_=smote_cd.dataset_generation.generate_dataset(n_features,n_classes,size_sample*20,betas=betas,random_state=i+9999)

        X_logratio,y_logratio=smote_cd.oversampling_multioutput(X_for_os,y_for_os,label_distance='logratio',n_iter_max=2e3,k=10,norm=2)
        X_compositional,y_compositional=smote_cd.oversampling_multioutput(X_for_os,y_for_os,label_distance='compositional',n_iter_max=2e3,k=10,norm=2)

        y_compositional_predict = softmax(gb_predict_compositional(X_compositional,logratio_transform(y_compositional),X_test))
        y_predict = softmax(gb_predict_raw(X,logratio_transform(y),X_test))
        y_logratio_predict = softmax(gb_predict_logratio(X_logratio,logratio_transform(y_logratio),X_test))

        r2_raw.append(r2_score(y_test, y_predict, multioutput='raw_values'))
        r2_compositional.append(r2_score(y_test, y_compositional_predict, multioutput='raw_values'))
        r2_logratio.append(r2_score(y_test, y_logratio_predict, multioutput='raw_values'))

        accuracy_raw.append(accuracy(y_test, y_predict))
        accuracy_compositional.append(accuracy(y_test, y_compositional_predict))
        accuracy_logratio.append(accuracy(y_test, y_logratio_predict))

        f1_raw.append(f1_score(np.argmax(y_test,axis=1),np.argmax(y_predict,axis=1),average=None))
        f1_compositional.append(f1_score(np.argmax(y_test,axis=1),np.argmax(y_compositional_predict,axis=1),average=None))
        f1_logratio.append(f1_score(np.argmax(y_test,axis=1),np.argmax(y_logratio_predict,axis=1),average=None))
        
    r2_compositional_tot = np.round(np.mean(r2_compositional,axis=0),5)
    r2_raw_tot = np.round(np.mean(r2_raw,axis=0),5)
    r2_logratio_tot = np.round(np.mean(r2_logratio,axis=0),5)
    accuracy_compositional_tot = np.round(np.mean(accuracy_compositional),5)
    accuracy_raw_tot = np.round(np.mean(accuracy_raw),5)
    accuracy_logratio_tot = np.round(np.mean(accuracy_logratio),5)
    f1_compositional_tot = np.round(np.mean(f1_compositional,axis=0),5)
    f1_raw_tot = np.round(np.mean(f1_raw,axis=0),5)
    f1_logratio_tot = np.round(np.mean(f1_logratio,axis=0),5)
    imbalance_ratio_tot = np.round(np.mean(imbalance_ratio),5)

    r2_compositional_std = np.round(np.std(r2_compositional,axis=0),5)
    r2_raw_std = np.round(np.std(r2_raw,axis=0),5)
    r2_logratio_std = np.round(np.std(r2_logratio,axis=0),5)
    accuracy_compositional_std = np.round(np.std(accuracy_compositional),5)
    accuracy_raw_std = np.round(np.std(accuracy_raw),5)
    accuracy_logratio_std = np.round(np.std(accuracy_logratio),5)
    f1_compositional_std = np.round(np.std(f1_compositional,axis=0),5)
    f1_raw_std = np.round(np.std(f1_raw,axis=0),5)
    f1_logratio_std = np.round(np.std(f1_logratio,axis=0),5)
    imbalance_ratio_std = np.round(np.std(imbalance_ratio),5)
        
    return(r2_compositional_tot, r2_raw_tot, r2_logratio_tot,
           accuracy_compositional_tot, accuracy_raw_tot, accuracy_logratio_tot,
           f1_compositional_tot, f1_raw_tot, f1_logratio_tot, 
           imbalance_ratio_tot,
           r2_compositional_std, r2_raw_std, r2_logratio_std,
           accuracy_compositional_std, accuracy_raw_std, accuracy_logratio_std,
           f1_compositional_std, f1_raw_std, f1_logratio_std, 
           imbalance_ratio_std)


def perf_evaluation_nn_repetition(j, n_features=n_features, n_classes=n_classes, size_sample=size_sample, 
                                         betas_init=betas_init, n_iter=100, verbose=False, step=1, undersampling=False):
    
    betas=np.copy(betas_init)
    betas[0,0]+=j
        
    r2_compositional=[]
    r2_raw=[]
    r2_logratio=[]
    accuracy_compositional=[]
    accuracy_raw=[]
    accuracy_logratio=[]
    f1_compositional=[]
    f1_raw=[]
    f1_logratio=[]

    imbalance_ratio=[]

    for i in range(n_iter):

        X,y,_=smote_cd.dataset_generation.generate_dataset(n_features,n_classes,size_sample,betas=betas,random_state=i)
        imbalance_ratio.append(sum(y)[0]/np.sum(y))

        if undersampling:
            undersampled_indexes = smote_cd.random_undersampling(y)
            indexes_to_keep = [i for i in range(len(y)) if i not in undersampled_indexes]
            y_for_os = y[indexes_to_keep]
            X_for_os = X[indexes_to_keep]
        else:
            y_for_os = np.copy(y)
            X_for_os = np.copy(X)

        X_test,y_test,_=smote_cd.dataset_generation.generate_dataset(n_features,n_classes,size_sample*20,betas=betas,random_state=i+9999)

        X_logratio,y_logratio=smote_cd.oversampling_multioutput(X_for_os,y_for_os,label_distance='logratio',n_iter_max=2e3,k=10,norm=2)
        X_compositional,y_compositional=smote_cd.oversampling_multioutput(X_for_os,y_for_os,label_distance='compositional',n_iter_max=2e3,k=10,norm=2)

        hyperparameters_raw = {'activation': 'identity', 'alpha': 1e-05, 'beta_1': 0.95, 'hidden_layer_sizes': (40,), 
                                   'learning_rate': 'constant', 'learning_rate_init': 0.001, 'max_iter': 10000, 
                                   'momentum': 0.9, 'random_state': 2, 'solver': 'sgd'}
        nn = MLPRegressor(**hyperparameters_raw)
        nn.fit(X, logratio_transform(y))
        y_predict = softmax(nn.predict(X_test))

        hyperparameters_nn_compositional = {'activation': 'logistic', 'alpha': 0.001, 'beta_1': 0.95, 
                                            'hidden_layer_sizes': (20,), 'learning_rate': 'constant', 
                                            'learning_rate_init': 0.0001, 'max_iter': 10000, 'momentum': 0.8, 
                                            'random_state': 2, 'solver': 'adam'}
        nn_compositional = MLPRegressor(**hyperparameters_nn_compositional)
        nn_compositional.fit(X_compositional, logratio_transform(y_compositional))
        y_compositional_predict = softmax(nn_compositional.predict(X_test))

        hyperparameters_nn_logratio = {'activation': 'identity', 'alpha': 0.001, 'beta_1': 0.9, 
                                       'hidden_layer_sizes': (80,), 'learning_rate': 'constant', 
                                       'learning_rate_init': 0.0001, 'max_iter': 10000, 'momentum': 0.9, 
                                       'random_state': 2, 'solver': 'adam'}
        nn_logratio = MLPRegressor(**hyperparameters_nn_logratio)
        nn_logratio.fit(X_logratio, logratio_transform(y_logratio))
        y_logratio_predict = softmax(nn_logratio.predict(X_test))

        r2_raw.append(r2_score(y_test, y_predict, multioutput='raw_values'))
        r2_compositional.append(r2_score(y_test, y_compositional_predict, multioutput='raw_values'))
        r2_logratio.append(r2_score(y_test, y_logratio_predict, multioutput='raw_values'))

        accuracy_raw.append(accuracy(y_test, y_predict))
        accuracy_compositional.append(accuracy(y_test, y_compositional_predict))
        accuracy_logratio.append(accuracy(y_test, y_logratio_predict))

        f1_raw.append(f1_score(np.argmax(y_test,axis=1),np.argmax(y_predict,axis=1),average=None))
        f1_compositional.append(f1_score(np.argmax(y_test,axis=1),np.argmax(y_compositional_predict,axis=1),average=None))
        f1_logratio.append(f1_score(np.argmax(y_test,axis=1),np.argmax(y_logratio_predict,axis=1),average=None))
        
    r2_compositional_tot = np.round(np.mean(r2_compositional,axis=0),5)
    r2_raw_tot = np.round(np.mean(r2_raw,axis=0),5)
    r2_logratio_tot = np.round(np.mean(r2_logratio,axis=0),5)
    accuracy_compositional_tot = np.round(np.mean(accuracy_compositional),5)
    accuracy_raw_tot = np.round(np.mean(accuracy_raw),5)
    accuracy_logratio_tot = np.round(np.mean(accuracy_logratio),5)
    f1_compositional_tot = np.round(np.mean(f1_compositional,axis=0),5)
    f1_raw_tot = np.round(np.mean(f1_raw,axis=0),5)
    f1_logratio_tot = np.round(np.mean(f1_logratio,axis=0),5)
    imbalance_ratio_tot = np.round(np.mean(imbalance_ratio),5)

    r2_compositional_std = np.round(np.std(r2_compositional,axis=0),5)
    r2_raw_std = np.round(np.std(r2_raw,axis=0),5)
    r2_logratio_std = np.round(np.std(r2_logratio,axis=0),5)
    accuracy_compositional_std = np.round(np.std(accuracy_compositional),5)
    accuracy_raw_std = np.round(np.std(accuracy_raw),5)
    accuracy_logratio_std = np.round(np.std(accuracy_logratio),5)
    f1_compositional_std = np.round(np.std(f1_compositional,axis=0),5)
    f1_raw_std = np.round(np.std(f1_raw,axis=0),5)
    f1_logratio_std = np.round(np.std(f1_logratio,axis=0),5)
    imbalance_ratio_std = np.round(np.std(imbalance_ratio),5)
        
    return(r2_compositional_tot, r2_raw_tot, r2_logratio_tot,
           accuracy_compositional_tot, accuracy_raw_tot, accuracy_logratio_tot,
           f1_compositional_tot, f1_raw_tot, f1_logratio_tot, 
           imbalance_ratio_tot,
           r2_compositional_std, r2_raw_std, r2_logratio_std,
           accuracy_compositional_std, accuracy_raw_std, accuracy_logratio_std,
           f1_compositional_std, f1_raw_std, f1_logratio_std, 
           imbalance_ratio_std)


# Comparison between imbalance and performance 

def perf_evaluation(n_features, n_classes, size_sample, betas_init, method, n_imbalanced_points=12, n_iter=50, verbose=False, step=1, undersampling=False):

    r2_compositional_tot=[]
    r2_raw_tot=[]
    r2_logratio_tot=[]
    accuracy_compositional_tot=[]
    accuracy_raw_tot=[]
    accuracy_logratio_tot=[]
    f1_compositional_tot=[]
    f1_raw_tot=[]
    f1_logratio_tot=[]
    imbalance_ratio_tot=[]
    
    r2_compositional_std=[]
    r2_raw_std=[]
    r2_logratio_std=[]
    accuracy_compositional_std=[]
    accuracy_raw_std=[]
    accuracy_logratio_std=[]
    f1_compositional_std=[]
    f1_raw_std=[]
    f1_logratio_std=[]
    
    imbalance_ratio_std=[]

    params=[]
    
    for j in range(0,n_imbalanced_points,step):
        params.append(j)
        
    pool = Pool(processes=6)
    if method=='dirichlet':
        outputs = pool.map(perf_evaluation_dirichlet_repetition, params)
    elif method=='gb':
        outputs = pool.map(perf_evaluation_gb_repetition, params)
    elif method=='nn':
        outputs = pool.map(perf_evaluation_nn_repetition, params)
    else:
        print('ERROR : method unknown')
        outputs = None
    
    return outputs



##### PARALLEL COMPUTING #####

# Comparison between imbalance and performance 

def perf_evaluation_gb_parallel(i, n_features, n_classes, size_sample, betas_init, n_imbalanced_points=12, n_iter=100, verbose=False, step=1, undersampling=False):

    r2_compositional_tot, r2_raw_tot, r2_logratio_tot = [],[],[]
    crossentropy_compositional_tot, crossentropy_raw_tot, crossentropy_logratio_tot = [], [], []
    rmse_compositional_tot, rmse_raw_tot, rmse_logratio_tot = [], [], []
    accuracy_compositional_tot, accuracy_raw_tot, accuracy_logratio_tot = [], [], []
    f1_compositional_tot, f1_raw_tot, f1_logratio_tot = [], [], []
    imbalance_ratio_tot=[]
    
    r2_compositional_std, r2_raw_std, r2_logratio_std = [], [], []
    crossentropy_compositional_std, crossentropy_raw_std, crossentropy_logratio_std = [], [], []
    rmse_compositional_std, rmse_raw_std, rmse_logratio_std = [], [], []
    accuracy_compositional_std, accuracy_raw_std, accuracy_logratio_std = [], [], []
    f1_compositional_std, f1_raw_std, f1_logratio_std = [], [], []
    imbalance_ratio_std=[]

    for j in range(0,n_imbalanced_points,step):
        
        if verbose:
            print(j)

        betas=np.copy(betas_init)
        betas[0,0]+=j

        r2_compositional, r2_raw, r2_logratio = [], [], []
        crossentropy_compositional, crossentropy_raw, crossentropy_logratio = [], [], []
        rmse_compositional, rmse_raw, rmse_logratio = [], [], []
        accuracy_compositional, accuracy_raw, accuracy_logratio = [], [], []
        f1_compositional, f1_raw, f1_logratio = [], [], []
        

        imbalance_ratio=[]
        
        X,y,_=smote_cd.dataset_generation.generate_dataset(n_features,n_classes,size_sample,betas=betas,random_state=i)
        imbalance_ratio.append(sum(y)[0]/np.sum(y))

        if undersampling:
            undersampled_indexes = smote_cd.random_undersampling(y)
            indexes_to_keep = [i for i in range(len(y)) if i not in undersampled_indexes]
            y_for_os = y[indexes_to_keep]
            X_for_os = X[indexes_to_keep]
        else:
            y_for_os = np.copy(y)
            X_for_os = np.copy(X)

        X_test,y_test,_=smote_cd.dataset_generation.generate_dataset(n_features,n_classes,size_sample*20,betas=betas,random_state=i+9999)

        X_logratio,y_logratio=smote_cd.oversampling_multioutput(X_for_os,y_for_os,label_distance='logratio',n_iter_max=2e3,k=10,norm=2)
        X_compositional,y_compositional=smote_cd.oversampling_multioutput(X_for_os,y_for_os,label_distance='compositional',n_iter_max=2e3,k=10,norm=2)

        y_compositional_predict = softmax(gb_predict_compositional(X_compositional,logratio_transform(y_compositional),X_test))
        y_predict = softmax(gb_predict_raw(X,logratio_transform(y),X_test))
        y_logratio_predict = softmax(gb_predict_logratio(X_logratio,logratio_transform(y_logratio),X_test))

        r2_raw.append(r2_score(y_test, y_predict, multioutput='raw_values'))
        r2_compositional.append(r2_score(y_test, y_compositional_predict, multioutput='raw_values'))
        r2_logratio.append(r2_score(y_test, y_logratio_predict, multioutput='raw_values'))

        crossentropy_raw.append(crossentropy(y_test, y_predict))
        crossentropy_compositional.append(crossentropy(y_test, y_compositional_predict))
        crossentropy_logratio.append(crossentropy(y_test, y_logratio_predict))

        accuracy_raw.append(accuracy(y_test, y_predict))
        accuracy_compositional.append(accuracy(y_test, y_compositional_predict))
        accuracy_logratio.append(accuracy(y_test, y_logratio_predict))

        f1_raw.append(f1_score(np.argmax(y_test,axis=1),np.argmax(y_predict,axis=1),average=None))
        f1_compositional.append(f1_score(np.argmax(y_test,axis=1),np.argmax(y_compositional_predict,axis=1),average=None))
        f1_logratio.append(f1_score(np.argmax(y_test,axis=1),np.argmax(y_logratio_predict,axis=1),average=None))

        rmse_raw.append(mean_squared_error(y_test, y_predict, squared=False))
        rmse_compositional.append(mean_squared_error(y_test, y_compositional_predict, squared=False))
        rmse_logratio.append(mean_squared_error(y_test, y_logratio_predict, squared=False))
        
    return(r2_compositional, r2_raw, r2_logratio,
           crossentropy_compositional, crossentropy_raw, crossentropy_logratio,
           rmse_compositional, rmse_raw, rmse_logratio,
           accuracy_compositional, accuracy_raw, accuracy_logratio,
           f1_compositional, f1_raw, f1_logratio,
           imbalance_ratio)


def perf_evaluation_nn_parallel(i, n_features, n_classes, size_sample, betas_init, n_imbalanced_points=12, n_iter=100, verbose=False, step=1, undersampling=False):

    r2_compositional_tot, r2_raw_tot, r2_logratio_tot = [],[],[]
    crossentropy_compositional_tot, crossentropy_raw_tot, crossentropy_logratio_tot = [], [], []
    rmse_compositional_tot, rmse_raw_tot, rmse_logratio_tot = [], [], []
    accuracy_compositional_tot, accuracy_raw_tot, accuracy_logratio_tot = [], [], []
    f1_compositional_tot, f1_raw_tot, f1_logratio_tot = [], [], []
    imbalance_ratio_tot=[]
    
    r2_compositional_std, r2_raw_std, r2_logratio_std = [], [], []
    crossentropy_compositional_std, crossentropy_raw_std, crossentropy_logratio_std = [], [], []
    rmse_compositional_std, rmse_raw_std, rmse_logratio_std = [], [], []
    accuracy_compositional_std, accuracy_raw_std, accuracy_logratio_std = [], [], []
    f1_compositional_std, f1_raw_std, f1_logratio_std = [], [], []
    imbalance_ratio_std=[]

    for j in range(0,n_imbalanced_points,step):
        
        if verbose:
            print(j)

        betas=np.copy(betas_init)
        betas[0,0]+=j

        r2_compositional, r2_raw, r2_logratio = [], [], []
        crossentropy_compositional, crossentropy_raw, crossentropy_logratio = [], [], []
        rmse_compositional, rmse_raw, rmse_logratio = [], [], []
        accuracy_compositional, accuracy_raw, accuracy_logratio = [], [], []
        f1_compositional, f1_raw, f1_logratio = [], [], []
        

        imbalance_ratio=[]
        
        X,y,_=smote_cd.dataset_generation.generate_dataset(n_features,n_classes,size_sample,betas=betas,random_state=i)
        imbalance_ratio.append(sum(y)[0]/np.sum(y))

        if undersampling:
            undersampled_indexes = smote_cd.random_undersampling(y)
            indexes_to_keep = [i for i in range(len(y)) if i not in undersampled_indexes]
            y_for_os = y[indexes_to_keep]
            X_for_os = X[indexes_to_keep]
        else:
            y_for_os = np.copy(y)
            X_for_os = np.copy(X)

        X_test,y_test,_=smote_cd.dataset_generation.generate_dataset(n_features,n_classes,size_sample*20,betas=betas,random_state=i+9999)
            
        X_logratio,y_logratio=smote_cd.oversampling_multioutput(X_for_os,y_for_os,label_distance='logratio',n_iter_max=2e3,k=10,norm=2)

        X_compositional,y_compositional=smote_cd.oversampling_multioutput(X_for_os,y_for_os,label_distance='compositional',n_iter_max=2e3,k=10,norm=2)

        hyperparameters_raw = {'activation': 'identity', 'alpha': 1e-05, 'beta_1': 0.95, 'hidden_layer_sizes': (40,), 
                               'learning_rate': 'constant', 'learning_rate_init': 0.001, 'max_iter': 10000, 
                               'momentum': 0.9, 'random_state': 2, 'solver': 'sgd'}
        nn = MLPRegressor(**hyperparameters_raw)
        nn.fit(X, logratio_transform(y))
        y_predict = softmax(nn.predict(X_test))

        hyperparameters_nn_compositional = {'activation': 'logistic', 'alpha': 0.001, 'beta_1': 0.95, 
                                            'hidden_layer_sizes': (20,), 'learning_rate': 'constant', 
                                            'learning_rate_init': 0.0001, 'max_iter': 10000, 'momentum': 0.8, 
                                            'random_state': 2, 'solver': 'adam'}
        nn_compositional = MLPRegressor(**hyperparameters_nn_compositional)
        nn_compositional.fit(X_compositional, logratio_transform(y_compositional))
        y_compositional_predict = softmax(nn_compositional.predict(X_test))

        hyperparameters_nn_logratio = {'activation': 'identity', 'alpha': 0.001, 'beta_1': 0.9, 
                                       'hidden_layer_sizes': (80,), 'learning_rate': 'constant', 
                                       'learning_rate_init': 0.0001, 'max_iter': 10000, 'momentum': 0.9, 
                                       'random_state': 2, 'solver': 'adam'}
        nn_logratio = MLPRegressor(**hyperparameters_nn_logratio)
        nn_logratio.fit(X_logratio, logratio_transform(y_logratio))
        y_logratio_predict = softmax(nn_logratio.predict(X_test))

        r2_raw.append(r2_score(y_test, y_predict, multioutput='raw_values'))
        r2_compositional.append(r2_score(y_test, y_compositional_predict, multioutput='raw_values'))
        r2_logratio.append(r2_score(y_test, y_logratio_predict, multioutput='raw_values'))

        crossentropy_raw.append(crossentropy(y_test, y_predict))
        crossentropy_compositional.append(crossentropy(y_test, y_compositional_predict))
        crossentropy_logratio.append(crossentropy(y_test, y_logratio_predict))

        accuracy_raw.append(accuracy(y_test, y_predict))
        accuracy_compositional.append(accuracy(y_test, y_compositional_predict))
        accuracy_logratio.append(accuracy(y_test, y_logratio_predict))

        f1_raw.append(f1_score(np.argmax(y_test,axis=1),np.argmax(y_predict,axis=1),average=None))
        f1_compositional.append(f1_score(np.argmax(y_test,axis=1),np.argmax(y_compositional_predict,axis=1),average=None))
        f1_logratio.append(f1_score(np.argmax(y_test,axis=1),np.argmax(y_logratio_predict,axis=1),average=None))

        rmse_raw.append(mean_squared_error(y_test, y_predict, squared=False))
        rmse_compositional.append(mean_squared_error(y_test, y_compositional_predict, squared=False))
        rmse_logratio.append(mean_squared_error(y_test, y_logratio_predict, squared=False))
        
    return(r2_compositional, r2_raw, r2_logratio,
           crossentropy_compositional, crossentropy_raw, crossentropy_logratio,
           rmse_compositional, rmse_raw, rmse_logratio,
           accuracy_compositional, accuracy_raw, accuracy_logratio,
           f1_compositional, f1_raw, f1_logratio,
           imbalance_ratio)



def perf_evaluation_dirichlet_parallel(i, n_features, n_classes, size_sample, betas_init, n_imbalanced_points=12, n_iter=100, verbose=False, step=1, undersampling=False):

    r2_compositional_tot, r2_raw_tot, r2_logratio_tot = [],[],[]
    crossentropy_compositional_tot, crossentropy_raw_tot, crossentropy_logratio_tot = [], [], []
    rmse_compositional_tot, rmse_raw_tot, rmse_logratio_tot = [], [], []
    accuracy_compositional_tot, accuracy_raw_tot, accuracy_logratio_tot = [], [], []
    f1_compositional_tot, f1_raw_tot, f1_logratio_tot = [], [], []
    imbalance_ratio_tot=[]
    
    r2_compositional_std, r2_raw_std, r2_logratio_std = [], [], []
    crossentropy_compositional_std, crossentropy_raw_std, crossentropy_logratio_std = [], [], []
    rmse_compositional_std, rmse_raw_std, rmse_logratio_std = [], [], []
    accuracy_compositional_std, accuracy_raw_std, accuracy_logratio_std = [], [], []
    f1_compositional_std, f1_raw_std, f1_logratio_std = [], [], []
    imbalance_ratio_std=[]

    for j in range(0,n_imbalanced_points,step):
        
        if verbose:
            print(j)

        betas=np.copy(betas_init)
        betas[0,0]+=j

        r2_compositional, r2_raw, r2_logratio = [], [], []
        crossentropy_compositional, crossentropy_raw, crossentropy_logratio = [], [], []
        rmse_compositional, rmse_raw, rmse_logratio = [], [], []
        accuracy_compositional, accuracy_raw, accuracy_logratio = [], [], []
        f1_compositional, f1_raw, f1_logratio = [], [], []
        

        imbalance_ratio=[]
        
        X,y,_=smote_cd.dataset_generation.generate_dataset(n_features,n_classes,size_sample,betas=betas,random_state=i)
        imbalance_ratio.append(sum(y)[0]/np.sum(y))

        if undersampling:
            undersampled_indexes = smote_cd.random_undersampling(y)
            indexes_to_keep = [i for i in range(len(y)) if i not in undersampled_indexes]
            y_for_os = y[indexes_to_keep]
            X_for_os = X[indexes_to_keep]
        else:
            y_for_os = np.copy(y)
            X_for_os = np.copy(X)

        X_test,y_test,_=smote_cd.dataset_generation.generate_dataset(n_features,n_classes,size_sample*20,betas=betas,random_state=i+9999)
            
        X_logratio,y_logratio=smote_cd.oversampling_multioutput(X_for_os,y_for_os,label_distance='logratio',n_iter_max=2e3,k=10,norm=2)
        X_compositional,y_compositional=smote_cd.oversampling_multioutput(X_for_os,y_for_os,label_distance='compositional',n_iter_max=2e3,k=10,norm=2)


        _,y_compositional_predict=dirichlet_model(X_compositional, y_compositional, X_test, y_test, n_features)
        _,y_predict=dirichlet_model(X, y, X_test, y_test, n_features)
        _,y_logratio_predict=dirichlet_model(X_logratio, y_logratio, X_test, y_test, n_features)

        r2_raw.append(r2_score(y_test, y_predict, multioutput='raw_values'))
        r2_compositional.append(r2_score(y_test, y_compositional_predict, multioutput='raw_values'))
        r2_logratio.append(r2_score(y_test, y_logratio_predict, multioutput='raw_values'))

        crossentropy_raw.append(crossentropy(y_test, y_predict))
        crossentropy_compositional.append(crossentropy(y_test, y_compositional_predict))
        crossentropy_logratio.append(crossentropy(y_test, y_logratio_predict))

        accuracy_raw.append(accuracy(y_test, y_predict))
        accuracy_compositional.append(accuracy(y_test, y_compositional_predict))
        accuracy_logratio.append(accuracy(y_test, y_logratio_predict))

        f1_raw.append(f1_score(np.argmax(y_test,axis=1),np.argmax(y_predict,axis=1),average=None))
        f1_compositional.append(f1_score(np.argmax(y_test,axis=1),np.argmax(y_compositional_predict,axis=1),average=None))
        f1_logratio.append(f1_score(np.argmax(y_test,axis=1),np.argmax(y_logratio_predict,axis=1),average=None))

        rmse_raw.append(mean_squared_error(y_test, y_predict, squared=False))
        rmse_compositional.append(mean_squared_error(y_test, y_compositional_predict, squared=False))
        rmse_logratio.append(mean_squared_error(y_test, y_logratio_predict, squared=False))
        
    return(r2_compositional, r2_raw, r2_logratio,
           crossentropy_compositional, crossentropy_raw, crossentropy_logratio,
           rmse_compositional, rmse_raw, rmse_logratio,
           accuracy_compositional, accuracy_raw, accuracy_logratio,
           f1_compositional, f1_raw, f1_logratio,
           imbalance_ratio)




########################################
##### FOR THE REAL DATASETS

def eval_perf_gb(random_s, X, Y, k_folds=5):
    
    r2_compositional, r2_raw, r2_logratio = [], [], []
    crossentropy_compositional, crossentropy_raw, crossentropy_logratio = [], [], []
    rmse_compositional, rmse_raw, rmse_logratio = [], [], []
    accuracy_compositional, accuracy_raw, accuracy_logratio = [], [], []
    f1_compositional, f1_raw, f1_logratio = [], [], []
    
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_s)
    kf.get_n_splits(X)
    for train_index, test_index in kf.split(X):
        X_train, y_train = X[train_index], Y[train_index]
        X_test, y_test = X[test_index], Y[test_index]

        y_predict = gb_predict(X[train_index], y_train, X_test)

        X_train_logratio, y_train_logratio = smote_cd.oversampling_multioutput(X_train,y_train, label_distance='logratio')
        y_logratio_predict = gb_predict(X_train_logratio, y_train_logratio, X_test)

        X_train_compos, y_train_compos = smote_cd.oversampling_multioutput(X_train,y_train, label_distance='compositional')
        y_compositional_predict = gb_predict(X_train_compos, y_train_compos, X_test)

        r2_raw.append(r2_score(y_test, y_predict, multioutput='raw_values'))
        r2_compositional.append(r2_score(y_test, y_compositional_predict, multioutput='raw_values'))
        r2_logratio.append(r2_score(y_test, y_logratio_predict, multioutput='raw_values'))

        crossentropy_raw.append(crossentropy(y_test, y_predict))
        crossentropy_compositional.append(crossentropy(y_test, y_compositional_predict))
        crossentropy_logratio.append(crossentropy(y_test, y_logratio_predict))

        accuracy_raw.append(accuracy(y_test, y_predict))
        accuracy_compositional.append(accuracy(y_test, y_compositional_predict))
        accuracy_logratio.append(accuracy(y_test, y_logratio_predict))

        f1_raw.append(f1_score(np.argmax(y_test,axis=1),np.argmax(y_predict,axis=1),average=None))
        f1_compositional.append(f1_score(np.argmax(y_test,axis=1),np.argmax(y_compositional_predict,axis=1),average=None))
        f1_logratio.append(f1_score(np.argmax(y_test,axis=1),np.argmax(y_logratio_predict,axis=1),average=None))

        rmse_raw.append(mean_squared_error(y_test, y_predict, squared=False))
        rmse_compositional.append(mean_squared_error(y_test, y_compositional_predict, squared=False))
        rmse_logratio.append(mean_squared_error(y_test, y_logratio_predict, squared=False))

    return(r2_compositional, r2_raw, r2_logratio,
           crossentropy_compositional, crossentropy_raw, crossentropy_logratio,
           rmse_compositional, rmse_raw, rmse_logratio,
           accuracy_compositional, accuracy_raw, accuracy_logratio,
           f1_compositional, f1_raw, f1_logratio)
    
    
def eval_perf_nn(random_s, X, Y, k_folds=5):
    
    r2_compositional, r2_raw, r2_logratio = [], [], []
    crossentropy_compositional, crossentropy_raw, crossentropy_logratio = [], [], []
    rmse_compositional, rmse_raw, rmse_logratio = [], [], []
    accuracy_compositional, accuracy_raw, accuracy_logratio = [], [], []
    f1_compositional, f1_raw, f1_logratio = [], [], []
    
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_s)
    kf.get_n_splits(X)
    for train_index, test_index in kf.split(X):
        X_train, y_train = X[train_index], Y[train_index]
        X_test, y_test = X[test_index], Y[test_index]

        X_train_logratio, y_train_logratio = smote_cd.oversampling_multioutput(X_train,y_train, label_distance='logratio')

        X_train_compos, y_train_compos = smote_cd.oversampling_multioutput(X_train,y_train, label_distance='compositional')
        
        model_raw = Sequential()
        model_raw.add(Dense(70, input_shape = (100, )))
        model_raw.add(Activation('tanh'))
        model_raw.add(Dense(70))
        model_raw.add(Activation('tanh'))
        model_raw.add(Dense(70))
        model_raw.add(Activation('tanh'))
        model_raw.add(Dense(3))
        model_raw.add(Activation('softmax'))
        model_raw.compile(loss = 'categorical_crossentropy', optimizer='adam')
        model_raw.fit(X_train, y_train)
        y_predict = model_raw.predict(X_test)

        model_compositional = Sequential()
        model_compositional.add(Dense(70, input_shape = (100, )))
        model_compositional.add(Activation('tanh'))
        model_compositional.add(Dense(70))
        model_compositional.add(Activation('tanh'))
        model_compositional.add(Dense(70))
        model_compositional.add(Activation('tanh'))
        model_compositional.add(Dense(3))
        model_compositional.add(Activation('softmax'))
        model_compositional.compile(loss = 'categorical_crossentropy', optimizer='adam')
        model_compositional.fit(X_train_compos, y_train_compos)
        y_compositional_predict = model_compositional.predict(X_test)
        
        model_logratio = Sequential()
        model_logratio.add(Dense(70, input_shape = (100, )))
        model_logratio.add(Activation('tanh'))
        model_logratio.add(Dense(70))
        model_logratio.add(Activation('tanh'))
        model_logratio.add(Dense(70))
        model_logratio.add(Activation('tanh'))
        model_logratio.add(Dense(3))
        model_logratio.add(Activation('softmax'))
        model_logratio.compile(loss = 'categorical_crossentropy', optimizer='adam')
        model_logratio.fit(X_train_logratio, y_train_logratio)
        y_logratio_predict = model_logratio.predict(X_test)

        r2_raw.append(r2_score(y_test, y_predict, multioutput='raw_values'))
        r2_compositional.append(r2_score(y_test, y_compositional_predict, multioutput='raw_values'))
        r2_logratio.append(r2_score(y_test, y_logratio_predict, multioutput='raw_values'))

        crossentropy_raw.append(crossentropy(y_test, y_predict))
        crossentropy_compositional.append(crossentropy(y_test, y_compositional_predict))
        crossentropy_logratio.append(crossentropy(y_test, y_logratio_predict))

        accuracy_raw.append(accuracy(y_test, y_predict))
        accuracy_compositional.append(accuracy(y_test, y_compositional_predict))
        accuracy_logratio.append(accuracy(y_test, y_logratio_predict))

        f1_raw.append(f1_score(np.argmax(y_test,axis=1),np.argmax(y_predict,axis=1),average=None))
        f1_compositional.append(f1_score(np.argmax(y_test,axis=1),np.argmax(y_compositional_predict,axis=1),average=None))
        f1_logratio.append(f1_score(np.argmax(y_test,axis=1),np.argmax(y_logratio_predict,axis=1),average=None))

        rmse_raw.append(mean_squared_error(y_test, y_predict, squared=False))
        rmse_compositional.append(mean_squared_error(y_test, y_compositional_predict, squared=False))
        rmse_logratio.append(mean_squared_error(y_test, y_logratio_predict, squared=False))

    return(r2_compositional, r2_raw, r2_logratio,
           crossentropy_compositional, crossentropy_raw, crossentropy_logratio,
           rmse_compositional, rmse_raw, rmse_logratio,
           accuracy_compositional, accuracy_raw, accuracy_logratio,
           f1_compositional, f1_raw, f1_logratio)


def eval_perf_dirichlet(random_s, X, Y, k_folds=5):
    
    n_features = X.shape[1]
    
    r2_compositional, r2_raw, r2_logratio = [], [], []
    crossentropy_compositional, crossentropy_raw, crossentropy_logratio = [], [], []
    rmse_compositional, rmse_raw, rmse_logratio = [], [], []
    accuracy_compositional, accuracy_raw, accuracy_logratio = [], [], []
    f1_compositional, f1_raw, f1_logratio = [], [], []
    
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_s)
    kf.get_n_splits(X)
    for train_index, test_index in kf.split(X):
        X_train, y_train = X[train_index], Y[train_index]
        X_test, y_test = X[test_index], Y[test_index]
        
        X_train_logratio, y_train_logratio = smote_cd.oversampling_multioutput(X_train,y_train, label_distance='logratio')
        X_train_compos, y_train_compos = smote_cd.oversampling_multioutput(X_train,y_train, label_distance='compositional')
        
        _,y_compositional_predict=dirichlet_model(X_train_compos, y_train_compos, X_test, y_test, n_features)
        _,y_predict=dirichlet_model(X_train, y_train, X_test, y_test, n_features)
        _,y_logratio_predict=dirichlet_model(X_train_logratio, y_train_logratio, X_test, y_test, n_features)

        r2_raw.append(r2_score(y_test, y_predict, multioutput='raw_values'))
        r2_compositional.append(r2_score(y_test, y_compositional_predict, multioutput='raw_values'))
        r2_logratio.append(r2_score(y_test, y_logratio_predict, multioutput='raw_values'))

        crossentropy_raw.append(crossentropy(y_test, y_predict))
        crossentropy_compositional.append(crossentropy(y_test, y_compositional_predict))
        crossentropy_logratio.append(crossentropy(y_test, y_logratio_predict))

        accuracy_raw.append(accuracy(y_test, y_predict))
        accuracy_compositional.append(accuracy(y_test, y_compositional_predict))
        accuracy_logratio.append(accuracy(y_test, y_logratio_predict))

        f1_raw.append(f1_score(np.argmax(y_test,axis=1),np.argmax(y_predict,axis=1),average=None))
        f1_compositional.append(f1_score(np.argmax(y_test,axis=1),np.argmax(y_compositional_predict,axis=1),average=None))
        f1_logratio.append(f1_score(np.argmax(y_test,axis=1),np.argmax(y_logratio_predict,axis=1),average=None))

        rmse_raw.append(mean_squared_error(y_test, y_predict, squared=False))
        rmse_compositional.append(mean_squared_error(y_test, y_compositional_predict, squared=False))
        rmse_logratio.append(mean_squared_error(y_test, y_logratio_predict, squared=False))

    return(r2_compositional, r2_raw, r2_logratio,
           crossentropy_compositional, crossentropy_raw, crossentropy_logratio,
           rmse_compositional, rmse_raw, rmse_logratio,
           accuracy_compositional, accuracy_raw, accuracy_logratio,
           f1_compositional, f1_raw, f1_logratio)



########## FOR MAUPITI

def eval_perf_gb_maupiti(random_s, X, Y, k_folds=5):
    
    r2_compositional, r2_raw, r2_logratio = [], [], []
    crossentropy_compositional, crossentropy_raw, crossentropy_logratio = [], [], []
    rmse_compositional, rmse_raw, rmse_logratio = [], [], []
    accuracy_compositional, accuracy_raw, accuracy_logratio = [], [], []
    f1_compositional, f1_raw, f1_logratio = [], [], []
    
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_s)
    kf.get_n_splits(X)
    for train_index, test_index in kf.split(X):
        X_train, y_train = X[train_index], Y[train_index]
        X_test, y_test = X[test_index], Y[test_index]

        y_predict = softmax(gb_predict(X_train, logratio_transform(y_train), X_test))

        X_train_logratio, y_train_logratio = smote_cd.oversampling_multioutput(X_train,y_train, label_distance='logratio')
        y_logratio_predict = softmax(gb_predict(X_train_logratio, logratio_transform(y_train_logratio), X_test))

        X_train_compos, y_train_compos = smote_cd.oversampling_multioutput(X_train,y_train, label_distance='compositional')
        y_compositional_predict = softmax(gb_predict(X_train_compos, logratio_transform(y_train_compos), X_test))

        r2_raw.append(r2_score(y_test, y_predict, multioutput='raw_values'))
        r2_compositional.append(r2_score(y_test, y_compositional_predict, multioutput='raw_values'))
        r2_logratio.append(r2_score(y_test, y_logratio_predict, multioutput='raw_values'))

        crossentropy_raw.append(crossentropy(y_test, y_predict))
        crossentropy_compositional.append(crossentropy(y_test, y_compositional_predict))
        crossentropy_logratio.append(crossentropy(y_test, y_logratio_predict))

        accuracy_raw.append(accuracy(y_test, y_predict))
        accuracy_compositional.append(accuracy(y_test, y_compositional_predict))
        accuracy_logratio.append(accuracy(y_test, y_logratio_predict))

        f1_raw.append(f1_score(np.argmax(y_test,axis=1),np.argmax(y_predict,axis=1),average=None))
        f1_compositional.append(f1_score(np.argmax(y_test,axis=1),np.argmax(y_compositional_predict,axis=1),average=None))
        f1_logratio.append(f1_score(np.argmax(y_test,axis=1),np.argmax(y_logratio_predict,axis=1),average=None))

        rmse_raw.append(mean_squared_error(y_test, y_predict, squared=False))
        rmse_compositional.append(mean_squared_error(y_test, y_compositional_predict, squared=False))
        rmse_logratio.append(mean_squared_error(y_test, y_logratio_predict, squared=False))

    return(r2_compositional, r2_raw, r2_logratio,
           crossentropy_compositional, crossentropy_raw, crossentropy_logratio,
           rmse_compositional, rmse_raw, rmse_logratio,
           accuracy_compositional, accuracy_raw, accuracy_logratio,
           f1_compositional, f1_raw, f1_logratio)

def eval_perf_nn_maupiti(random_s, X, Y, k_folds=5):
    
    r2_compositional, r2_raw, r2_logratio = [], [], []
    crossentropy_compositional, crossentropy_raw, crossentropy_logratio = [], [], []
    rmse_compositional, rmse_raw, rmse_logratio = [], [], []
    accuracy_compositional, accuracy_raw, accuracy_logratio = [], [], []
    f1_compositional, f1_raw, f1_logratio = [], [], []
    
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_s)
    kf.get_n_splits(X)
    for train_index, test_index in kf.split(X):
        X_train, y_train = X[train_index], Y[train_index]
        X_test, y_test = X[test_index], Y[test_index]

        X_train_logratio, y_train_logratio = smote_cd.oversampling_multioutput(X_train,y_train, label_distance='logratio')

        X_train_compos, y_train_compos = smote_cd.oversampling_multioutput(X_train,y_train, label_distance='compositional')
        
        hyperparameters_raw = {'hidden_layer_sizes': (80, 40,), 'activation': 'relu', 
                               'max_iter': 500, 'random_state': 2, 'solver': 'adam'}
        nn = MLPRegressor(**hyperparameters_raw)
        nn.fit(X_train, logratio_transform(y_train))
        y_predict = softmax(nn.predict(X_test))

        hyperparameters_nn_compositional = {'hidden_layer_sizes': (80, 40,), 'activation': 'relu',
                                            'max_iter': 500, 'random_state': 2, 'solver': 'adam'}
        nn_compositional = MLPRegressor(**hyperparameters_nn_compositional)
        nn_compositional.fit(X_train_compos, logratio_transform(y_train_compos))
        y_compositional_predict = softmax(nn_compositional.predict(X_test))

        hyperparameters_nn_logratio = {'hidden_layer_sizes': (80, 40,), 'activation': 'relu',
                                       'max_iter': 500, 'random_state': 2, 'solver': 'adam'}
        nn_logratio = MLPRegressor(**hyperparameters_nn_logratio)
        nn_logratio.fit(X_train_logratio, logratio_transform(y_train_logratio))
        y_logratio_predict = softmax(nn_logratio.predict(X_test))

        r2_raw.append(r2_score(y_test, y_predict, multioutput='raw_values'))
        r2_compositional.append(r2_score(y_test, y_compositional_predict, multioutput='raw_values'))
        r2_logratio.append(r2_score(y_test, y_logratio_predict, multioutput='raw_values'))

        crossentropy_raw.append(crossentropy(y_test, y_predict))
        crossentropy_compositional.append(crossentropy(y_test, y_compositional_predict))
        crossentropy_logratio.append(crossentropy(y_test, y_logratio_predict))

        accuracy_raw.append(accuracy(y_test, y_predict))
        accuracy_compositional.append(accuracy(y_test, y_compositional_predict))
        accuracy_logratio.append(accuracy(y_test, y_logratio_predict))

        f1_raw.append(f1_score(np.argmax(y_test,axis=1),np.argmax(y_predict,axis=1),average=None))
        f1_compositional.append(f1_score(np.argmax(y_test,axis=1),np.argmax(y_compositional_predict,axis=1),average=None))
        f1_logratio.append(f1_score(np.argmax(y_test,axis=1),np.argmax(y_logratio_predict,axis=1),average=None))

        rmse_raw.append(mean_squared_error(y_test, y_predict, squared=False))
        rmse_compositional.append(mean_squared_error(y_test, y_compositional_predict, squared=False))
        rmse_logratio.append(mean_squared_error(y_test, y_logratio_predict, squared=False))

    return(r2_compositional, r2_raw, r2_logratio,
           crossentropy_compositional, crossentropy_raw, crossentropy_logratio,
           rmse_compositional, rmse_raw, rmse_logratio,
           accuracy_compositional, accuracy_raw, accuracy_logratio,
           f1_compositional, f1_raw, f1_logratio)