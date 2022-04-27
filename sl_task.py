import numpy as np
from abc import ABC, abstractmethod


class sl_task(ABC):
    def __init__(self, X, Y, family, cv_folds):
        self.data = {"X" = X, "Y" = Y}
        self.family = family
        self.cv_folds = cv_folds