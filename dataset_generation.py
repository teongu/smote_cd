# This file provides functions to generate a synthetic dataset with compositional labels.
# Author : Teo Nguyen

import numpy as np
import pandas as pd
import random

##############################################################################################################

def softmax(k,x):
    """ Returns the softmax function for the k-th value of x. """
    return(np.exp(x[k])/np.sum(np.exp(x)))

##############################################################################################################

def generate_betas(n_features,n_classes,random_state=None):
    """ Randomly generate a betas matrix. The size of the matrix will be (n_classes,n_features+1). """
    np.random.seed(random_state)
    betas=np.random.rand(n_classes,n_features+1)
    return(betas)

##############################################################################################################

def generate_dataset(n_features,n_classes,size,betas=None,random_state=None):
    """ Generate a synthetic dataset with compositional labels. 
    Input:
            - n_features (int) : The desired number of features for the dataset.
            - n_classes (int) : The desired number of classes for the dataset.
            - size (int) : The number of points to create in the dataset.
            - betas (ndarray; default=None) : The betas matrix used to generate the data. If None, a random one is created.
            - random_state (int; default=None) : The random seed to use.
    Output:
            - list_X (ndarray) : The array of the features of the created dataset.
            - list_y (ndarray) : The array of the labels of the created dataset.
            - betas (ndarray) : The betas matrix, either which has been set as an input or the created one.
    """
    # creation of betas vector
    if betas is None:
        np.random.seed(random_state)
        betas=np.random.rand(n_classes,n_features+1)
    
    # creation of probabilities and initialization
    np.random.seed(random_state)
    list_X=20*np.random.rand(size,n_features+1)-10
    list_X[:,0]=1
    list_y=[]
    
    # filling the features and labels
    for x in list_X:
        beta_dot_x=[np.dot(betas[i],x) for i in range(n_classes)]
        alphas=[softmax(i,beta_dot_x) for i in range(n_classes)]
        list_y.append(np.random.dirichlet(alphas))
        
    list_y=np.array(list_y)
    list_X=np.array(list_X)[:,1:]
    return(list_X,list_y,betas)
