"""
Example for creating a synthetic dataset with compositional labels.

Requirements
------------
seaborn
mpl_toolkits
"""

from smote_cd import dataset_generation

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

def display_data():
    
    n_features=2
    n_classes=2
    size_sample=400
    
    betas1 = np.array([[0.4, 0.2, 0.5],[0.4, 0.4, 0.3]])
    X1,y1,_=dataset_generation.generate_dataset(n_features,n_classes,size_sample,betas=betas1,random_state=2)

    betas2 = np.array([[0.1, 0, 0.8],[0.9, 0.5, 0.1]])
    X2,y2,_=dataset_generation.generate_dataset(n_features,n_classes,size_sample,betas=betas2,random_state=2)

    cmap = sns.color_palette("crest", as_cmap=True)

    f, ax = plt.subplots(1,2, figsize=(16,8))

    points = ax[0].scatter(np.transpose(X1)[0],np.transpose(X1)[1],c=y1[:,0],s=20, cmap=cmap)
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("left", size="3%", pad=0.5)
    cax.axis('off')
    ax[0].set_title("(a)", fontsize=18)

    points = ax[1].scatter(np.transpose(X2)[0],np.transpose(X2)[1],c=y2[:,0],s=20, cmap=cmap)
    divider2 = make_axes_locatable(ax[1])
    cax2 = divider2.append_axes("left", size="3%", pad=0.5)
    ax[1].set_title("(b)",  fontsize=18)

    cbar = f.colorbar(points, cax=cax2, orientation='vertical')

    cbar.ax.set_title("Value of class 0")

    ax[0].tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False)
    ax[1].tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False)

    plt.show()
    
    
if __name__ == '__main__':
    
    display_data()