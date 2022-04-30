
from code.learners import Learner
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_regression

# A screener is a learner that train with Y and X, then
# chain the selected features to the next learner in the pipeline, and
# predict the selected features of new input X.

class Screener(Learner):
    def __init__(self, sl_task: sl_task, learners: list, name = None, params = None):
        super().__init__(name = name, params = params)
        self.learners = learners

    def train(self, Y: np.ndarray, X: np.ndarray) -> None:
        """train learners sequentially
        
        Parameters
        ----------
        Y  response varible / labels
        X  features

        Returns
        -------
        None
        """
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """predict with new X
        
        Parameters
        ----------
        X (new) input features for prediction

        Returns
        -------
        predictons
        """
        pass

    def chain(self) -> None:
        # In general, we can even chain pipelines.
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


class Screen_univar(Screener):
    # select k features based on univariate statistical tests
    def __init__(self, sl_task: sl_task, k: int, name = None, params = None):
        super().__init__(name = name, params = params)
        self.k = k
        self.X_star = None
        self.i_star = None

    def train(self, Y: np.ndarray, X: np.ndarray) -> None:
        """train learners sequentially
        
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
            stats_test = f_regression
        elif (family == "Binomial"):
            stats_test = chi2
        
        screener = SelectKBest(stats_test, k = self.k)
        screener.fit(X, Y)
        idx = screener.get_support(indices = True)

        # save
        self.X_star = X[:, idx]
        self.i_star = idx

    def predict(self, X: np.ndarray) -> np.ndarray:
        """predict with new X
        
        Parameters
        ----------
        X (new) input features for prediction

        Returns
        -------
        predictons
        """
        # assume X has the same shape and order of features 
        # as the original X
        self.X_star = X[:, self.i_star]
        return self.X_star

    def chain(self) -> None:
        """output for next learner
        
        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        return self.X_star


