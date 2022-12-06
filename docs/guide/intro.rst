.. _intro:

############
Introduction
############

The details of this work are presented in the paper Nguyen et al. 2023 "SMOTE for compositional data".

Compositional data
==================

Given an integer :math:`D`, that will here represent the number of classes, we define a :math:`D`-part compositional data as a vector :math:`x = [x_1, x_2, \dots, x_D]` such that:

.. math::

    \left\{ 
    \begin{array}{l} 
    \forall i \in [1,2,\dots,D], x_i \geq 0 \\
    \displaystyle \sum_{i=1}^D x_i = 1.  \end{array}\right.

And a simplex :math:`S^D` is defined as the ensemble of all the :math:`D`-part compositional data, i.e.

.. math::

    S^D = \left\{ x = [x_1, x_2, ..., x_D] \ |\ \forall i \in [1,2,\dots,D], x_i \geq 0 ; \sum_{i=1}^D x_i = 1  \right\}.

Imbalance
=========

In our case, we consider that the labels :math:`y` of our dataset are compositional, and that the dataset has :math:`n` entries. The balance of a dataset is measured with a vector representing the percentage of the sums for each class, i.e. a vector :math:`V = [ y^\Sigma_{(1)}, y^\Sigma_{(2)}, \dots, y^\Sigma_{(D)} ] / \sum_{j=1}^D y^\Sigma_{(j)}`, where :math:`y^\Sigma_{(j)} = \sum_{i=1}^n y_{ij}` is the sum of all the values of class :math:`j`. With this definition, the dataset is perfectly balanced if the vector of the sums is :math:`[ 1/D, \dots, 1/D]`.

The package ``smote_cd`` deals with imbalance in dataset having compositional data labels and real-values features. Note that it could also be used when the features are compositional and the labels are real-values (by interverting the two terms), but the performances would not be as expected because the nearest neighbors would be computed based on the labels, and not features, which is not exactly how the algorithm is supposed to be designed.
