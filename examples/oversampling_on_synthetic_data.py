"""
Example of the SMOTE for compositional data, applied on synthetic data generated with 
the function smote_cd.dataset_generation.generate_dataset, having 2 features and 2 classes.
"""

import numpy as np
import matplotlib.pyplot as plt
import smote_cd


def create_imbalanced_dataset():
    
    n_features=2
    n_classes=2
    size_sample=500
    betas = np.array([[.4, .9, .2], [.9, .5, .6]])
    
    X,y,_ = smote_cd.dataset_generation.generate_dataset(n_features,n_classes,size_sample,betas=betas,random_state=4)
    
    # remove points of one the classes, keeping only 20 points, to simulate a high imbalance
    y = np.concatenate((y[np.argmax(X,axis=1)==0][:20],y[np.argmax(X,axis=1)==1]))
    X = np.concatenate((X[np.argmax(X,axis=1)==0][:20],X[np.argmax(X,axis=1)==1]))
    
    return(X,y)


def oversampling_and_display(X,y):
    
    X_os,y_os=smote_cd.oversampling_multioutput(X,y,n_iter_max=2e3,k=10,norm=2)
    
    fig, ax = plt.subplots(1, 2, figsize=(16, 8), sharex=True)

    ax[0].scatter(X[:,0],X[:,1],s=30,c=y[:,0], marker='o')
    ax[0].set_title("(a)")

    ax[1].scatter(X[:,0],X[:,1],s=30,c=y[:,0], marker='o', label='Original data')
    ax[1].scatter(X_os[len(X):,0],X_os[len(X):,1],c=y_os[len(X):,0],s=30, marker='x', label='Synthetic data')
    ax[1].set_title("(b)")

    plt.legend(prop={'size': 14})

    plt.show()
    
    
if __name__ == '__main__':
    
    X,y = create_imbalanced_dataset()
    oversampling_and_display(X,y)
    