# This file provides functions to perform the oversampling and undersampling of compositional data.
# Author : Teo Nguyen

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gmean
from sklearn.preprocessing import StandardScaler


##############################################################################################################

def distance_simplex(point1,point2):
    """ 
    Compute the distance in the simplex between two vectors point1 and point2. 
    
    Parameters
    ---------
    point1 : array_like, shape (n,)
        Vector to compute the distance.
    point2 : array_like, shape (n,)
        Vector to compute the distance.
        
    Returns
    -------
    float
        The distance (in the simplex) between point1 and point2. 
    """
    # copy the points to replace the 0
    x=np.copy(point1)
    y=np.copy(point2)
    x[x==0]=1e-20
    y[y==0]=1e-20
    # return the distance
    return( np.sqrt( sum( ( np.log(x/gmean(x))-np.log(y/gmean(y)) )**2 ) ) )


##############################################################################################################

def create_logratio_point(p1,p2,w1,w2):
    """ 
    Create a new point in the simplex by using the logratio transform. 
    
    Parameters
    ----------
    p1 : array_like, shape (n,)
        First point to create a new point.
    p2 : array_like, shape (n,)
        Second point to create a new point.
    w1 : float 
        Weight associated to the first point ``0 <= w1 <= l``.
    w2 : float
        Weight associated to the second point ``w2 = 1 - w1``.
    
    Returns
    -------
    numpy.ndarray, shape (n,)
        The new point created in the simplex.
    """
    lrp1=np.copy(np.array(p1,dtype='float'))
    lrp2=np.copy(np.array(p2,dtype='float'))
    # transform from the euclidian space to the logratio
    lrp1[lrp1==0]=1e-20
    lrp2[lrp2==0]=1e-20
    lrp1=np.log(lrp1/gmean(lrp1))
    lrp2=np.log(lrp2/gmean(lrp2))
    new_point=w1*lrp1+w2*lrp2
    # transform from the logratio to the euclidian space
    new_point=np.exp(new_point)/sum(np.exp(new_point))
    new_point[new_point<1e-19]=0
    return(new_point)


##############################################################################################################
# These are the operations from Wang 2015 "Principal component analysis for compositional data vectors" 
# https://link.springer.com/article/10.1007/s00180-015-0570-1

def closure(x):
    """ 
    Equation 6 from Wang 2015 "Principal component analysis for compositional data vectors".
    
    Parameters
    ----------
    x : array_like, shape (n,)
        The vector on which the closure operator is performed.
    
    Returns
    -------
    numpy.ndarray, shape (n,)
        The closure vector of x.
    """
    return(np.array(x)/np.sum(x))

def perturbation(x,y):
    """ 
    Equation 4 from Wang 2015 "Principal component analysis for compositional data vectors".
    
    Parameters
    ----------
    x : array_like, shape (n,)
        The vector on which the closure operator is performed.
    y : array_like shape (n,)
        The vector on which the perturbation operator is performed.
    
    Returns
    -------
    numpy.ndarray, shape (n,)
        The perturbation operator between x and y.
    """
    return(closure(np.array(x)*np.array(y)))

def power(beta,x):
    """ 
    Equation 5 from Wang 2015 "Principal component analysis for compositional data vectors".
    
    Parameters
    ----------
    beta : float
        The scalar to perform the power operator.
    x : array_like, shape (n,)
        The vector to perform the power operator.
    
    Returns
    -------
    numpy.ndarray, shape (n,)
        The power vector between x and beta.
    """
    return(closure(np.power(x,beta)))

def create_new_point(p1,p2,w):
    """
    Create a new point in the simplex by using the compositional distance operations in the simplex. 
    
    Parameters
    ----------
    p1 : array_like, shape (n,)
        First point to create a new point.
    p2 : array_like, shape (n,)
        Second point to create a new point.
    w : float 
        Weight associated to the first point ``0 <= w <= l``.
    
    Returns
    -------
    numpy.ndarray, shape (n,)
        The new created point with the compositional distance operations.
    """
    # Corresponds to w*p1 + (1-w)*p2
    return(perturbation(power(w,p1),power(1-w,p2)))



##############################################################################################################

def oversampling_multioutput(df_features,df_labels,label_distance='logratio',normalize=False,
                             k=5,n_iter_max=100,norm=2,verbose=0,choice_new_point='min'):
    """ 
    Perform the oversampling on data which has a compositional label. 
    
    Parameters
    ----------
    df_features : array_like, shape (n,k)
        The features (X) of the data to be oversampled.
    df_labels : array_like, shape (n,q)
        The labels (y) of the data to be oversampled.
    label_distance : {'compositional', 'euclidian', 'logratio'}, optional
        The distance to be used to compute the label of the new point based on two existing points and a random weight 
        (the default is 'logratio').
    normalize : bool, optional
        Whether to normalize the features at the beggining of the algorithm (the default is False).
    k : int, optional
        The number of nearest neighbors among which a random neighbor is chosen (the default is 5).
    n_iter_max : int, optional
        The maximum number of iterations to be performed (the default is 100).
    norm : {non-zero int, inf, -inf, 'fro', 'nuc'}, optional
        The order of the norm used to compute the nearest neighbors (the default is 2).
    verbose : int or bool, optional
        Whether to print text detailing the steps of the algorithm (the default is 0).
    choice_new_point : {'min', 'random'}, optional
        How a new point is selected : randomly or in the smallest class (the default is 'min').
    
    Returns
    -------
    features : numpy.ndarray, shape (n+m,k)
        The oversampled features, containing the old ones (first n values) and the created ones (last m values).
    labels : numpy.ndarray, shape (n+m,q)
        The oversampled labels, containing the old ones (first n values) and the created ones (last m values).
    """
    
    # Checking the types
    if (type(df_features)==pd.DataFrame) | (type(df_features)==pd.Series) :
        features=df_features.values
    elif (type(df_features)==np.ndarray) | (type(df_features)==list):
        features=np.copy(df_features)
    else:
        raise Exception("Wrong input type for df_features. It should be a pd.DataFame, pd.Series, np.array or list.")
    
    # If needed, normalization of the features
    if normalize:
        features_for_knn = StandardScaler().fit_transform(features)
    else:
        features_for_knn = features
        
    if (type(df_labels)==pd.DataFrame) | (type(df_labels)==pd.Series) :
        labels=df_labels.values
    elif (type(df_labels)==np.ndarray) | (type(df_labels)==list) :
        labels=np.copy(df_labels)
    else:
        raise Exception("Wrong input type for df_labels. It should be a pd.DataFame, pd.Series, np.array or list.")

    n,q=np.shape(labels)
    
    # Computing the vector of the sum of the probabilities
    v_sum=sum(labels)
    
    # Computing the threshold and the majority class
    threshold=max(v_sum)
    ind_threshold=np.argmax(v_sum)
    
    # Creating the vector containing the classes that have reached the threshold
    vec_reach_threshold=[ind_threshold]
    vec_not_reach=[i for i in range(q)]
    del vec_not_reach[ind_threshold]
    v_sum_not_reach=list(np.copy(v_sum))
    del v_sum_not_reach[ind_threshold]
    v_sum_not_reach=np.array(v_sum_not_reach)
    
    # Creating the vector containing the indexes of the points of which the majority class have not reached the threshold
    # This is the indexes of the points that can be used for the creation of a new point
    ind_to_keep=[i for i in range(n) if np.argmax(labels[i]) in vec_not_reach]
    n_iter=0
    
    # These two elements are useful to avoid computing at each iteration the elements of the minority class
    # if the minority has not changed between two iterations
    class_min_previous = np.argmin(v_sum)
    elements_class_min = [i for i in ind_to_keep if (np.argmax(labels[i])==class_min_previous)]
    class_min_to_take=0
    # [1] If no point has the smallest class as its dominant class, then we take the second smallest class, and so on.
    while len(elements_class_min)==0:
        class_min_to_take+=1
        if class_min_to_take>=len(v_sum):
            raise Exception("Not enough points to duplicate.")
        class_min=v_sum.argsort()[class_min_to_take]
        elements_class_min=[i for i in ind_to_keep if (np.argmax(labels[i])==class_min)]
      
    # flag to know if the smallest class has been removed because it did not have enough points
    removed_class_min = False
    
    # In each iteration of the loop, a new point is created.
    # The loop stops when n_iter_max has been reached, 
    # or when all the classes have reached the threshold, 
    # or when we do not have enough points to create a new point
    while n_iter<n_iter_max and vec_not_reach!=[] and len(ind_to_keep)>k:
        
        n_iter+=1
        
        if choice_new_point=='random':
            # The point is randomly chosen
            r=int(random.choice(ind_to_keep))
        elif choice_new_point=='min':
            # The point is chosen among the smallest class
            # We find all the points of the smallest class
            class_min=np.argmin(v_sum) 
            # We only need to compute them if the smallest class has changed between two iterations,
            # or if the smallest class has been removed because there was not enough points
            if ((class_min != class_min_previous) or removed_class_min) :
                elements_class_min=[i for i in ind_to_keep if (np.argmax(labels[i])==class_min)]
                class_min_to_take=0
                # [1] If no point has the smallest class as its dominant class, then we take the second smallest class, and so on.
                while len(elements_class_min)==0:
                    class_min_to_take+=1
                    if class_min_to_take>=len(v_sum):
                        raise Exception("Not enough points to duplicate.")
                    class_min=v_sum.argsort()[class_min_to_take]
                    elements_class_min=[i for i in ind_to_keep if (np.argmax(labels[i])==class_min)]
                class_min_previous = class_min
                removed_class_min = False
            # The point is chosen among the points of the smallest class
            r=int(random.choice(elements_class_min))
        else:
            raise Exception("The method to choose a new point is unknown. It should be 'random' or 'min'.")
            
        # Defining the dominant class of the selected point, its labels and features.
        dominant_class_r=np.argmax(labels[r])
        base_point_lab=labels[r]
        base_point_feat=features[r]
        base_point_feat_for_knn=features_for_knn[r]
        
        # Finding the points that have the same dominant class than the selected point
        k_n=np.empty((k,2))
        ind_to_use=[i for i in ind_to_keep if ((i!=r) & (np.argmax(labels[i])==dominant_class_r))]
        
        # If there is not enough neighbors that have the same dominant class,
        # all the points having this dominant class are removed from the list of points that can be duplicated. 
        # It will end up in this class having no point in it, hence the utility of [1]
        if len(ind_to_use)<k:
            ind_to_keep.remove(r)
            ind_to_keep = [i for i in ind_to_keep if i not in ind_to_use]
            removed_class_min = True
            if verbose:
                print('Removed the points of class {} because there is not enough points to oversample.'.format(dominant_class_r))
            continue
        
        # initialisation to compute the k-nn
        k_n = np.array([ [i,np.linalg.norm(base_point_feat_for_knn-features_for_knn[i],norm)] for i in ind_to_use[0:k] ])
        m=max(k_n[:,1])
        # filling : finding the k-nn
        for i in ind_to_use[k:]:
            d=np.linalg.norm(base_point_feat_for_knn-features_for_knn[i],norm)
            if d<m:
                k_n[np.argmax(k_n[:,1]),:]=[i,d]
                m=max(k_n[:,1])

        # Randomly choose a point among the k-nn
        r_2=random.randint(0,k-1)
        random_neighboor_lab=labels[int(k_n[r_2,0])]
        random_neighboor_feat=features[int(k_n[r_2,0])]

        # Randomly choose a weight between 0 and 1 and compute the features and labels of the new point
        w=random.random()
        newpoint_feat=w*random_neighboor_feat+(1-w)*base_point_feat
        if label_distance=='euclidian':
            newpoint_lab=w*random_neighboor_lab+(1-w)*base_point_lab
        elif label_distance=='logratio':
            newpoint_lab=create_logratio_point(random_neighboor_lab,base_point_lab,w,1-w)
        elif label_distance=='compositional':
            newpoint_lab=create_new_point(random_neighboor_lab,base_point_lab,w)
        else:
            raise Exception("The distance to compute the label of the new points is not correct. It should be 'compositional', 'euclidian' or 'logratio'.")
        
        # Adding the new created point to the dataframes
        features=np.append(features,[newpoint_feat],axis=0)
        labels=np.append(labels,[newpoint_lab],axis=0)
        
        # Actualization of the sum vectors, and the vectors of the classes that reached the threshold.
        v_sum_not_reach=np.add(v_sum_not_reach,newpoint_lab[vec_not_reach])
        v_sum=np.add(v_sum,newpoint_lab)
        if any(item>threshold for item in v_sum_not_reach):
            ind_reach=np.where(v_sum_not_reach>threshold)[0]
            vec_reach_threshold=sorted(np.append(vec_reach_threshold,np.array(vec_not_reach)[ind_reach]))
            vec_not_reach = [i for i in vec_not_reach if i not in np.array(vec_not_reach)[ind_reach]]
            v_sum_not_reach=v_sum[vec_not_reach]
            ind_to_keep=[i for i in ind_to_keep if np.argmax(labels[i]) in vec_not_reach]           
    
    # Final message depending on the convergence condition
    if verbose:
        if vec_not_reach==[]:
            print('All classes have reached the threshold')
        elif n_iter==n_iter_max:
            print('Maximum number of iterations reached :',int(n_iter_max))
            print("Classes {} haven't reached the threshold".format(vec_not_reach))
        else :
            print("Classes {} haven't reached the threshold because there is not enough points to oversample".format(vec_not_reach))
        print('New points :',n_iter)
    
    return(features,labels)

##############################################################################################################

def undersampling(y,method='majority'):
    """ Performs a random undersampling.
    INPUT :
        - y (ndarray) : Array containing the compositional labels of the dataset to be undersampled. 
        - method ({'majority','all'}; default='majority') : If 'majority', undersamples only the majority class. If 'all', undersamples all classes except the minority.
    OUTPUT :
        - list_removed_indexes (list) : A list containing the indexes of the elements to be removed. 
    """
    n,q=np.shape(y)
    v_sum=sum(y)
    if method=='majority':
        i_max=np.argmax(v_sum)
        threshold=v_sum[np.argsort(v_sum)[-2]] # the second highest value
        indices_majority=[i for i in range(n) if np.argmax(y[i])==i_max]
    elif method=='all':
        i_min=np.argmin(v_sum)
        threshold=v_sum[i_min] # the smallest value
        indices_majority_all=[ [i for i in range(n) if np.argmax(y[i])==j ] for j in range(q) ]
        #indices_majority=[i for i in range(n) if np.argmax(y[i])!=i_min]
    else:
        raise Exception("The undersampling method is unknown.")
    list_removed_indexes=[]
    while np.max(v_sum)>threshold:
        
        if method=='all':
            indices_majority = indices_majority_all[np.argmax(v_sum)]
            i_to_remove=random.choice(indices_majority_all[np.argmax(v_sum)])
            indices_majority_all[np.argmax(v_sum)] = np.delete(indices_majority_all[np.argmax(v_sum)],
                                                               np.where(indices_majority==i_to_remove))
            
        else:
            i_to_remove=random.choice(indices_majority)
            indices_majority=np.delete(indices_majority,np.where(indices_majority==i_to_remove))
            
        v_sum=v_sum-y[i_to_remove]
        list_removed_indexes.append(i_to_remove)
        
        # update threshold
        if method=='all':
            threshold = v_sum[i_min]

    return(list_removed_indexes)