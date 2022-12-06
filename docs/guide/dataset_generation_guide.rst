.. _dataset_generation_guide:

##################
Dataset generation
##################

The ``smote_cd`` package comes with functions to generate synthetic datasets having compositional labels. The dataset is created using a Dirichlet distribution of parameter :math:`\alpha = X \beta`, :math:`X` being the features of the dataset and :math:`\beta` the matrix of the regression coefficients. 

Such a matrix :math:`\beta` can be created with the function :meth:`smote_cd.dataset_generation.generate_betas`:

.. code-block:: python

    betas = smote_cd.dataset_generation.generate_betas(n_features,n_classes)
    
The shape of the matrix :math:`\beta` is ``(n_classes, n_features+1)`` because the first column is associated to the intercept. Then, knowing such a matrix, one can generate a synthetic dataset of size ``size``:

.. code-block:: python

    X, y, _ = smote_cd.dataset_generation.generate_dataset(n_features, n_classes, size, betas=betas)

The third argument returned by the function is actually the :math:`\beta` matrix. Indeed, if no parameter ``betas`` is given, the function will create a new matrix :math:`\beta`. Of course, if you want reproducible results or to generate several datasets with the same distribution, it is better to first generate your matrix and then give it as an input of the function. 

Another way to keep the same ``betas`` is to get the one returned by the function to reuse it:

.. code-block:: python

    X_1, y_1, betas = smote_cd.dataset_generation.generate_dataset(n_features, n_classes, size_1)
    X_2, y_2, _ = smote_cd.dataset_generation.generate_dataset(n_features, n_classes, size_2, betas=betas)
    
Note that the parameter ``random_state`` will set the random seed to its given value. If no ``betas`` is specified, its creation will also be fixed by the random seed. It means that the two following calls return the same result:

.. code-block:: python

    X_1, y_1, _ = smote_cd.dataset_generation.generate_dataset(n_features, n_classes, size, random_seed=0)
    X_2, y_2, _ = smote_cd.dataset_generation.generate_dataset(n_features, n_classes, size, random_seed=0)

But these two do not return the same result:

.. code-block:: python

    X_1, y_1, _ = smote_cd.dataset_generation.generate_dataset(n_features, n_classes, size, betas=betas_1, random_seed=0)
    X_2, y_2, _ = smote_cd.dataset_generation.generate_dataset(n_features, n_classes, size, betas=betas_2, random_seed=0)
    
More example are available on the page :meth:`smote_cd.dataset_generation.generate_dataset`.