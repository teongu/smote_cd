# This file provides functions to generate a synthetic dataset with compositional labels.
# Author : Teo Nguyen

import numpy as np
import pandas as pd
import random

##############################################################################################################

def softmax(k,x):
    """ 
    Return the softmax function for the k-th value of the vector x. 
    
    Parameters
    ----------
    k : int
        The index at which the softmax is computed.
    x : array_like, shape (n,)
        The vector for which the softmax is computed.
    
    Returns
    -------
    float
        Value of the softmax of the k-th value of x.
    """
    return(np.exp(x[k])/np.sum(np.exp(x)))

##############################################################################################################

def generate_betas(n_features,n_classes,random_state=None):
    """ 
    Randomly generate a betas matrix of the regression coefficients. 
    
    Parameters
    ----------
    n_features : int
        The number of desired features.
    n_classes : int
        The number of desired classes.
    random_state : int, optional
        The random state for the generation.
        
    Returns
    -------
    array_like, shape (n_classes, n_features+1)
        The generated matrix. 
    """
    np.random.seed(random_state)
    betas=np.random.rand(n_classes,n_features+1)
    return(betas)

##############################################################################################################

def generate_dataset(n_features,n_classes,size,betas=None,random_state=None):
    """ 
    Generate a synthetic dataset with compositional labels. 
    
    Parameters
    ----------
    n_features : int
        The desired number of features for the dataset.
    n_classes : int
        The desired number of classes for the dataset.
    size : int
        The number of points to create in the dataset.
    betas : array_like, shape (n_classes, n_features+1), optional
        The betas matrix used to generate the data. If None, a random one is created.
    random_state : int, optional
        The random seed to use.
        
    Returns
    -------
    X : numpy.ndarray, shape (size, n_features)
        The array of the features of the created dataset.
    y : numpy.ndarray, shape (size, n_classes) 
        The array of the labels of the created dataset.
    betas : numpy.ndarray, shape (n_classes, n_features+1)
        The betas matrix, either the created one or the one set as an input.
    """
    # creation of betas vector
    if betas is None:
        np.random.seed(random_state)
        betas=np.random.rand(n_classes,n_features+1)
    
    # initialization
    np.random.seed(random_state)
    X=20*np.random.rand(size,n_features+1)-10
    X[:,0]=1
    y=[]
    
    # filling the features and labels
    for x in X:
        beta_dot_x=[np.dot(betas[i],x) for i in range(n_classes)]
        alphas=[softmax(i,beta_dot_x) for i in range(n_classes)]
        y.append(np.random.dirichlet(alphas))
        
    y=np.array(y)
    X=np.array(X)[:,1:]
    return(X,y,betas)
