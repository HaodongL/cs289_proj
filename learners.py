

import numpy as np
from abc import ABC, abstractmethod
import statsmodels.api as sm


class Learner(ABC):
    """Abstract class defining the common interface for all Learner methods."""
    def __init__(self, name = None, params = None, *args):
        self.name = name
        self.params = params
        self.fit_object = None,
        self.sl_task = None,

    @abstractmethod
    def train(self, Z):
        pass

    @abstractmethod
    def predict(self, Z):
        pass




class Lrnr_glm(Learner):
    def __init__(self):
        super().__init__()
        family = self.sl_task.family
        if (family == "Gaussian"):
            self.family = sm.families.Gaussian()
        elif (family == "Binomial"):
            self.family = sm.families.Binomial()

    def train(self, Y: np.ndarray, X: np.ndarray) -> None:
        """Fit model with training set
        
        Parameters
        ----------
        Y  response varible / labels
        X  features

        Returns
        -------
        None
        """
        model = sm.GLM(Y, X, family = self.family)
        self.fit_object = model.fit()


    def predict(self, X: np.ndarray) -> np.ndarray:
        """Backward pass for f(z) = z.
        
        Parameters
        ----------
        X (new) input features for prediction

        Returns
        -------
        predictons
        """
        fit = self.fit_object
        preds = fit.predict(X)
        return preds

