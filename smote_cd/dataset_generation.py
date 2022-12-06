# This file provides functions to generate a synthetic dataset with compositional labels.
# Author : Teo Nguyen

import numpy as np
import pandas as pd
import random

##############################################################################################################

def softmax(k,x):
    r""" 
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
    Randomly generate a betas matrix of the regression coefficients, that will be used to generate the dataset. 
    
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
        
    Notes
    -----
    The shape of the returned betas matrix is ``(n_classes, n_features+1)`` because its first column corresponds
    to the intercept.
        
    Examples
    --------
    With 2 features and 3 classes, the returned matrix will be of dimension (3,3).
    
    >>> from smote_cd import dataset_generation
    >>> dataset_generation.generate_betas(n_features=2,n_classes=3,random_state=0)
    array([[0.5488135 , 0.71518937, 0.60276338],
           [0.54488318, 0.4236548 , 0.64589411],
           [0.43758721, 0.891773  , 0.96366276]])
    
    With 3 features and 2 classes, the returned matrix will be of dimension (2,4).
    
    >>> dataset_generation.generate_betas(n_features=3,n_classes=2,random_state=0)
    array([[0.5488135 , 0.71518937, 0.60276338, 0.54488318],
           [0.4236548 , 0.64589411, 0.43758721, 0.891773  ]])
    
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
        
    Notes
    -----
    Each feature is uniformly generated between [-10, 10]. 
    The label at a given index, where the features are :math:`(x_1, \dots, x_p)`, is generated following a Dirichlet distribution 
    of parameter :math:`\\alpha`, where :math:`\\alpha` is:
    
    .. math:: \\alpha = \mbox{softmax} (B_{0,1} + B_{1,1} x_1 + \dots + B_{p,1} x_p, \dots, B_{0,K} + B_{1,K} x_1 + \dots + B_{p,K} x_p),
    
    where :math:`B` denotes the matrix ``beta``.
    
    
    Examples
    --------
    
    >>> from smote_cd import dataset_generation
    
    If ``betas`` is not provided, a matrix ``betas`` is created. As the matrix is created with the seed ``random_state``,
    if this parameter is specified, the created matrix will always be the same, and the points created aswell.  
    
    >>> X,_,betas = dataset_generation.generate_dataset(n_features=1, n_classes=2, size=5, random_state=0)
    >>> print(X)
    [[ 4.30378733]
     [ 0.89766366]
     [ 2.91788226]
     [ 7.83546002]
     [-2.33116962]]
    >>> print(betas)
    [[0.5488135  0.71518937]
     [0.60276338 0.54488318]]
    
    An common usage is to set a matrix ``betas`` to be able to randomly generate as many times as wanted, but always with 
    the same distribution. The following code will return 10 random points that will always follow the same distribution
    at each call:
    
    >>> betas = dataset_generation.generate_betas(n_features=1, n_classes=2,random_state=0)
    >>> X, y, _ = dataset_generation.generate_dataset(n_features=1, n_classes=2, betas=betas, size=10)
    
    However, the following code will return 10 random points that will not follow the same distribution at each call:
    
    >>> X, y, _ = dataset_generation.generate_dataset(n_features=1, n_classes=2, size=10)
    
    
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
