import numpy as np
from .bbo_agent import BBOAgent

from typing import Callable


class FCHC(BBOAgent):
    """
    First-choice hill-climbing (FCHC) for policy search is a black box optimization (BBO)
    algorithm. This implementation is a variant of Russell et al., 2003. It has 
    remarkably fewer hyperparameters than CEM, which makes it easier to apply. 
    
    Parameters
    ----------
    sigma (float): exploration parameter 
    theta (np.ndarray): initial mean policy parameter vector
    numEpisodes (int): the number of episodes to sample per policy
    evaluationFunction (function): evaluates a provided policy.
        input: policy (np.ndarray: a parameterized policy), numEpisodes
        output: the estimated return of the policy 
    """
    
    def __init__(self, theta:np.ndarray, sigma:float, evaluationFunction:Callable, numEpisodes:int=10):
        self._name = "First_Choice_Hill_Climbing"

        self._theta = theta  # TODO: set this value to the current mean parameter vector
        self._Sigma = sigma  # TODO: set this value to the current covariance matrix
        self.t = theta
        self.s = sigma
        self.numEpisodes = numEpisodes
        self.evaluationFunction = evaluationFunction
        self.J = evaluationFunction(self._theta,self.numEpisodes)
        # print(theta)
        # print(sigma)

    @property
    def name(self)->str:
        return self._name
    
    @property
    def parameters(self)->np.ndarray:
        return self._theta

    def train(self)->np.ndarray:
        theta_ = np.random.multivariate_normal(self._theta, self._Sigma * np.eye(self._theta.shape[0]))
        J_ = self.evaluationFunction(theta_,self.numEpisodes)
        if J_ > self.J:
            self._theta = theta_
            self.J = J_
        return self._theta
        # print("========")
        # print(self.J)
        # count = 4
        # while count>0:
        #     count-=1
        #     theta_ = np.random.multivariate_normal(self._theta, self._Sigma * np.eye(self._theta.shape[0]))
        #     print("===theta_===")
        #     print(theta_)
        #     print("++J_++")
        #     J_ = self.evaluationFunction(theta_,self.numEpisodes)
        #     print(J_)
        #     if J_>self.J:
        #         self._theta = theta_
        #         self.J = J_
        # print(self._theta)


    def reset(self)->None:
        self._theta = self.t
        self._Sigma = self.s
        self.J = self.evaluationFunction(self._theta,self.numEpisodes)