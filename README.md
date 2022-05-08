Replicating sl3 R package in Python

Key components of this SL:

    1. Learner options:
        a). glm (ols for cont; logistic for discrete)
        b). glmnet (lasso, ridge or elasticnet, support both cont and discrete Y)
        c). random forest (support both cont and discrete Y)
        d). xgboost (support both cont and discrete Y)

    2. Data process
        a). impute missing values
        b). convert categorical to dummy
        c). standardization

    3. Cross-validation scheme
        K-fold CV

    4. Loss function
        a). square error loss
        b). entropy loss

    5. Meta learner
        a). discrete (select the single best learner)
        b). nnls (non-negative least square)
        c). solnp (Nonlinear optimization using augmented Lagrange method)

    6. Pipeline
        A pipeline is a learner compose two or more learners.
        So the output of a learner in the pipeline will be
        the input of the next learner.

    7. Screener (with pipeline)
        a). univariate test 
        b). L1 penalization

    8. Variable Importance
        rank features based on the change of cv risks before and after permuting each feature.