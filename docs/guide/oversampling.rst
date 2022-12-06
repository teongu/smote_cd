.. _oversampling:

############
Oversampling
############

The oversampling on compositional data is performed with the function :meth:`smote_cd.oversampling_multioutput`.

The oversampling is an extension of the SMOTE technique, applied to compositional data. The detailed description and tests of the algorithm are given in the paper *Nguyen et al. 2023 "SMOTE for compositional data"*.

The main usage is the following : if you have a dataset with the features ``X`` of size ``(n,K)`` and the labels ``Y``  of size ``(n,J)``, you can use ``smote_cd`` if ``Y`` is compositional, i.e. if every row of ``Y`` represents a proportion that sums up to 1. This can be written as ``np.unique(np.sum(Y,axis=1))==1``. In that case, the call to use the oversampling is:

.. code-block:: python

    X_oversampled, y_oversampled = smote_cd.oversampling_multioutput(X,y)
   
In some cases, if the :math:`p` biggest classes are too large, you may want to perform an undersampling (:meth:`smote_cd.random_undersampling`) on them before applying the oversampling:

.. code-block:: python

    indexes_to_remove = smote_cd.random_undersampling(y)
    y_undersampled=np.delete(y,indexes_to_remove,axis=0)
    X_undersampled=np.delete(X,indexes_to_remove,axis=0)
    X_oversampled, y_oversampled = smote_cd.oversampling_multioutput(X_undersampled,y_undersampled)
    
Practical examples are available on the page :meth:`smote_cd.oversampling_multioutput`.