.. _undersampling:

#############
Undersampling
#############

In the case where ``method = 'majority'``, the random undersampling consists in randomly removing points which have the biggest class as their majority class. In other words, let :math:`M` be the biggest class of the dataset, i.e. :math:`M = \mbox{argmax}_j (V_j)`, where :math:`V` is the vector of the sum of the classes (as defined in :ref:`intro`). Then, all the points where the softmax of their label is :math:`M` are considered as points that can be removed. Some points are randomly removed one by one until :math:`V_M \leq V_{M_2}`, where :math:`M_2` is the second biggest class of the dataset.

In the case where ``method`` is an integer ``b``, then the ``b`` majority classes are similarly undersampled until the sum of each of them is :math:`\leq V_{M_{b+1}}`.

As the function ``smote_cd.random_undersampling`` only returns the indexes of the points to be removed, you will have to remove these indexes from your original dataset. For instance, you can do the following:

.. code-block:: python

    indexes_to_remove = smote_cd.random_undersampling(y)
    y_undersampled=np.delete(y,indexes_to_remove,axis=0)
    X_undersampled=np.delete(X,indexes_to_remove,axis=0)
    
Practical examples are available on the page :meth:`smote_cd.random_undersampling`.