

import numpy as np
from abc import ABC, abstractmethod
import statsmodels.api as sm
from sklearn.model_selection import KFold
from scipy.optimize import nnls


class Learner(ABC):
    """abstract class defining the common interface for all Learner methods."""
    def __init__(self, sl_task, name = None, params = None, *args):
        self.name = name
        self.params = params
        self.sl_task = sl_task
        self.fit_object = None
        

    @abstractmethod
    def train(self, Z):
        pass

    @abstractmethod
    def predict(self, Z):
        pass


def initialize_sl(stack, meta, name = None, params = None):

    return Lrnr_sl(
        stack = stack,
        meta = meta,
        name = name,
        params = params
    )


def square_error_loss(y_hat, y):
    return np.sum((y_hat - y)^2)/len(y)

def binomial_loglik_loss(y_hat, y):
    return -np.sum(y*np.log(y_hat) + (1 - y)*np.log(1-y_hat))/len(y)


class Lrnr_sl(Learner):
    def __init__(self, sl_task: sl_task, stack: list, meta: str , name = None, params = None):
        super().__init__(name = name, params = params)
        self.stack = stack
        self.folds = list(self.sl_task.cv_folds.split(self.sl_task.data["Y"]))
        self.n = len(self.sl_task.data["Y"])
        self.n_l = len(stack)
        self.n_k = len(folds)
        self.preds = np.zeros((self.n, self.n_l))
        self.fit_object_list = []
        self.meta = meta
        self.wt_meta = np.zeros((1, self.n_l))[0]
        self.cv_risk = None
        self.is_trained = False

        if (self.sl_task.family == 'Gaussian'):
            self.loss_f = square_error_loss
        elif (self.sl_task.family == 'Binomial'):
            self.loss_f = binomial_loglik_loss

    def train(self) -> None:
        """ 1. evaluate learners by cv risk, save
            2. discrete or meta learning, save
            3. fit the learners on full data, save
        
        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if (self.is_trained):
            raise Exception("This SL is already trained")

        # Step 1. calc cv risk by fitting each learner on t and preds on v
        for k in range(self.n_k):
            print("Training on Fold", k+1, "of", n_k, "\n")
            idx_t = self.folds[k][0]
            idx_v = self.folds[k][1]
            X_t = self.sl_task.data["X"][idx_t]
            Y_t = self.sl_task.data["Y"][idx_t]
            X_v = self.sl_task.data["X"][idx_v]
            Y_v = self.sl_task.data["Y"][idx_v]

            for l in range(self.n_l):
                lrnr = self.stack[l]
                lrnr.train(Y_t, X_t)
                preds = lrnr.predict(X_v)
                # save preds 
                self.preds[idx_v, l] = preds

        # save risk
        X = self.sl_task.data["X"]
        Y = self.sl_task.data["Y"].ravel()
        cv_risk = [self.loss_f(self.preds[:, l], Y) for l in range(self.n_l)]
        self.cv_risk = cv_risk

        # Step 2. discrete or meta learning, save
        if (self.meta == "discrete"):
            l_star = np.argmin(cv_risk)
            self.wt_meta[l_star] = 1
        elif (self.meta == "nnls"):
            res_meta = nnls(self.preds, Y)
            self.wt_meta = res_meta[0]

        # Step 3. fit learners on full data, save
        for l in range(self.n_l):
                lrnr = self.stack[l]
                lrnr.train(Y, X)

        self.is_trained = True

    def predict(self, X = None) -> np.ndarray:
        """predict with new X
        
        Parameters
        ----------
        X (new) input features for prediction

        Returns
        -------
        predictons
        """
        if (X is None):
            X = self.sl_task.data["X"]

        preds = np.zeros((len(X), self.n_l))

        for l in range(self.n_l):
            lrnr = self.stack[l]
            preds[:, l] = lrnr.predict(X)

        wt = self.wt_meta.reshape((1,self.n_l))
        preds =  np.sum(preds*wt, axis = 1)

        return preds


class Lrnr_glm(Learner):
    def __init__(self, sl_task: sl_task, name = None, params = None):
        super().__init__(name = name, params = params)
        family = self.sl_task.family
        if (family == "Gaussian"):
            self.family = sm.families.Gaussian()
        elif (family == "Binomial"):
            self.family = sm.families.Binomial()

    def train(self, Y: np.ndarray, X: np.ndarray) -> None:
        """fit model with training set
        
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
        """predict with new X
        
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