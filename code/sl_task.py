import numpy as np
from abc import ABC, abstractmethod
from sklearn.model_selection import KFold


class sl_task(ABC):
    def __init__(self, X, Y, family, cv_folds: KFold):
        self.data = {"X": X, "Y": Y}
        self.family = family
        self.cv_folds = cv_folds