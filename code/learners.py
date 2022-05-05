
from sl_task import sl_task
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from sklearn.model_selection import KFold
from scipy.optimize import nnls
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import pysolnp
import xgboost as xgb



class Learner(ABC):
    """abstract class defining the common interface for all Learner methods."""
    def __init__(self, sl_task, name = None, params = None, *args):
        self.name = name
        self.params = params
        self.sl_task = sl_task
        self.fit_object = None
        
    @abstractmethod
    def train(self, Y: np.ndarray, X: np.ndarray):
        pass

    @abstractmethod
    def chain(self):
        pass

    @abstractmethod
    def predict(self, X: np.ndarray):
        pass


def initialize_sl(sl_task, stack, meta, name = None, params = None):

    return Lrnr_sl(
        sl_task = sl_task,
        stack = stack,
        meta = meta,
        name = name,
        params = params
    )


def square_error_loss(y_hat, y):
    return np.sum((y_hat - y)**2)/len(y)

def binomial_loglik_loss(y_hat, y):
    return -np.sum(y*np.log(y_hat) + (1 - y)*np.log(1-y_hat))/len(y)


class Lrnr_sl(Learner):
    def __init__(self, sl_task: sl_task, stack: list, meta: str , name = None, params = None):
        super().__init__(sl_task = sl_task, name = name, params = params)
        self.stack = stack
        self.folds = list(self.sl_task.cv_folds.split(self.sl_task.data["Y"]))
        self.n = len(self.sl_task.data["Y"])
        self.p = self.sl_task.data["X"].shape[1]
        self.n_l = len(stack)
        self.n_k = len(self.folds)
        self.preds = np.zeros((self.n, self.n_l))
        self.meta = meta
        self.wt_meta = np.zeros((1, self.n_l))[0]
        self.cv_risk = None
        self.is_trained = False

        if (self.sl_task.family == 'Gaussian'):
            self.loss_f = square_error_loss
        elif (self.sl_task.family == 'Binomial'):
            self.loss_f = binomial_loglik_loss

        lrnr_names = []
        for lrnr in stack:
            lrnr_names.append(lrnr.name)
        self.lrnr_names = lrnr_names

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
            print("Training on Fold", k+1, "of", self.n_k, "\n")
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

        elif (self.meta == "solnp"):
            Z = self.preds
            # for simplicity, we just use linear
            # can be non-linear in general
            meta_linear = lambda b: Z @ b 
            obj_func = lambda b: self.loss_f(meta_linear(b), Y)
            eq_func = lambda b: [np.sum(b)]
            b0 = [1/self.n_l]*self.n_l
            bl = [0]*self.n_l
            bu = [1]*self.n_l
            eq_values = [1]

            res_meta = pysolnp.solve(
                obj_func = obj_func,
                par_start_value = b0,
                par_lower_limit = bl,
                par_upper_limit = bu,
                eq_func = eq_func,
                eq_values = eq_values)

            self.wt_meta = np.array(res_meta.optimum)


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

        wt = self.wt_meta
        preds =  preds @ wt

        return preds

    def chain(self) -> None:
        # For simplicity, skip this part for now.
        """output for next learner
        
        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        pass

    def summary(self) -> None:
        model_names = pd.DataFrame(self.lrnr_names)
        weights = pd.DataFrame(self.wt_meta)
        cv_risk = pd.DataFrame(self.cv_risk)

        tbl = pd.concat([model_names, weights, cv_risk], axis=1)
        tbl.columns = ['learner','weight','cv_risk']

        print("="*50)
        print(" Num. of cv folds: ", self.n_k, "\n", 
              "Meta learner: ", self.meta, "\n",
              "n: ", self.n, "\n",
              "p: ", self.p)
        print("="*50)
        print(tbl)


class Lrnr_glm(Learner):
    def __init__(self, sl_task: sl_task, name = "glm", params = None):
        super().__init__(sl_task = sl_task, name = name, params = params)
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

    def chain(self) -> None:
        # For simplicity, skip this part for now.
        """output for next learner
        
        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        pass


class Lrnr_glmnet(Learner):
    # Lasso when L1_wt = 1, ridge when L1_wt = 0, elasticnet when L1_wt = 0.5
    def __init__(self, sl_task: sl_task, name = "glmnet", params = [1, 1e-2]):
        super().__init__(sl_task = sl_task, name = name, params = params)
        self.L1_wt = params[0]
        self.Lambda = params[1]

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
        model = OLS(Y, X)
        self.fit_object = model.fit_regularized(alpha = self.Lambda, L1_wt = self.L1_wt)

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

    def chain(self) -> None:
        # For simplicity, skip this part for now.
        """output for next learner
        
        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        pass


class Lrnr_rf(Learner):
    def __init__(self, sl_task: sl_task, name = "rf", params = [3, None, 2]):
        super().__init__(sl_task = sl_task, name = name, params = params)
        self.max_depth = params[0]
        self.random_state = params[1]
        self.min_samples_split = params[2]
        

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
        family = self.sl_task.family
        if (family == "Binomial"):
            model = RandomForestClassifier(max_depth = self.max_depth, 
                                           random_state = self.random_state,
                                           min_samples_split = self.min_samples_split)
        elif (family == "Gaussian"):
            model = RandomForestRegressor(max_depth = self.max_depth, 
                                          random_state = self.random_state,
                                          min_samples_split = self.min_samples_split)

        self.fit_object = model.fit(X, Y)

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

    def chain(self) -> None:
        # For simplicity, skip this part for now.
        """output for next learner
        
        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        pass


class Lrnr_xgboost(Learner):
    def __init__(self, sl_task: sl_task, num_round: int, name = "xgb", params = [3, 1e-1, 100, 1, 1]):
        super().__init__(sl_task = sl_task, name = name, params = params)
        self.n_round = num_round
        self.max_depth = params[0]
        self.learning_rate = params[1]
        self.n_estimators = params[2]
        self.subsample = params[3]
        self.colsample_bytree = params[4]
        
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
        family = self.sl_task.family
        if (family == "Gaussian"):
            objective = "reg:squarederror"
        elif (family == "Binomial"):
            objective = "binary:logistic"

        dtrain = xgb.DMatrix(data = X,label = Y.ravel())

        xgb_params = {'objective': objective,
                      'learning_rate': self.learning_rate,
                      'max_depth': self.max_depth,
                      'n_estimators': self.n_estimators,
                      'subsample': self.subsample,
                      'colsample_bytree': self.colsample_bytree}

        self.fit_object = xgb.train(params = xgb_params, 
                                    dtrain = dtrain, 
                                    num_boost_round = self.n_round)

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
        dpred = xgb.DMatrix(data = X)
        preds = fit.predict(dpred)
        return preds

    def chain(self) -> None:
        # For simplicity, skip this part for now.
        """output for next learner
        
        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        pass