import numpy as np
from sklearn.model_selection import KFold


class sl_task:
    def __init__(self, X, Y, family, K: int):
        self.data = {"X": X, "Y": Y}
        self.family = family
        self.cv_folds = KFold(n_splits = K)