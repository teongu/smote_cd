"""
Example of the random undersampling for compositional data, applied on synthetic data generated with 
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


def undersampling_and_display(X,y):
    
    print('Sum of the classes before undersampling:',sum(y))
    
    indexes_to_remove = smote_cd.random_undersampling(y, method='majority')
    
    y_undersampled=np.delete(y,indexes_to_remove,axis=0)
    X_undersampled=np.delete(X,indexes_to_remove,axis=0)
    
    plt.scatter(X_undersampled[:,0],X_undersampled[:,1],c=y_undersampled[:,0],s=10)
    plt.show()
    
    print('Sum of the classes after undersampling:',sum(y_undersampled))
    
    
if __name__ == '__main__':
    
    X,y = create_imbalanced_dataset()
    undersampling_and_display(X,y)
    