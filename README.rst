###########################################################################
SMOTE-CD : Synthetic Minority Oversampling TEchnique for Compositional Data
###########################################################################

SMOTE for Compositional Data is an adaptation of the well-known SMOTE algorithm 
for the case where the labels of the dataset are compositional data. The package
can also be used when the features are compositional data (but not when both are
at the same time), by swapping the features and labels when calling the function.

**Documentation:** https://smote-cd.readthedocs.io/

Installation
============

SMOTE-CD can be installed on `Python 3.7 or above <https://www.python.org>`_.

Dependencies
------------

The following Python packages are required by SMOTE-CD:

* `NumPy <https://www.numpy.org>`_,
* `SciPy <https://www.scipy.org>`_,
* `Scikit-learn <https://scikit-learn.org/stable/index.html>`_

User installation
-----------------

You can install SMOTE-CD using ``pip`` ::

    python -m pip install smote-cd
