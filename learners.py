

import numpy as np
from abc import ABC, abstractmethod
import statsmodels.api as sm
from sklearn.model_selection import KFold


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


def initialize_sl(stack, name = None, params = None):

    return Lrnr_sl(
        stack = stack,
        name = name,
        params = params
    )



class Lrnr_sl(Learner):
    def __init__(self, stack: list, name = None, params = None):
        super().__init__(name = name, params = params)
        self.stack = stack
        self.folds = list(self.sl_task.cv_folds.split(self.sl_task.data["Y"]))
        self.preds = None
        self.fit_object_list = []


    def train(self) -> None:
        """Fit each learner on each training set,
           save the predictions on each v set
        
        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        for k in range(len(self.folds)):
            idx_t = self.folds[k][0]
            idx_v = self.folds[k][1]
            X_t = self.sl_task.data["X"][idx_t]
            Y_t = self.sl_task.data["Y"][idx_t]
            X_v = self.sl_task.data["X"][idx_v]
            Y_v = self.sl_task.data["Y"][idx_v]

            for l in range(len(self.stack)):
                lrnr = self.stack[l]
                fit_lrnr = lrnr.train(Y_t, X_t, family = lrnr.family).fit()
                preds = fit_lrnr(X_v)






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


class Lrnr_glm(Learner):
    def __init__(self, name = None, params = None):
        super().__init__(name = name, params = params)
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