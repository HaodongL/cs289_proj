import numpy as np
from sl_task import sl_task
from code.learners import Learner

# A pipeline is a learner that chain two or more learners
# So the output of a learner in the pipeline will be
# the input of the next learner
# For simplicity, here we only implement a screener pipeline with two learners

class Pipeline(Learner):
    def __init__(self, sl_task: sl_task, learners: list, name = None, params = None):
        super().__init__(sl_task = sl_task, name = name, params = params)
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
        X0 = X
        n_l = len(learners)

        for i in range(n_l):
        	learner = learners[i]
        	learner.train(Y, X0)
        	X0 = learner.chain()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """predict with new X
        
        Parameters
        ----------
        X (new) input features for prediction

        Returns
        -------
        predictons
        """
        preds = X
        n_l = len(learners)

        for i in range(n_l):
        	learner = learners[i]
        	preds = learner.predict(preds)
        	
        return preds

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
