import numpy as np
from .bbo_agent import BBOAgent

from typing import Callable


class CEM(BBOAgent):
    """
    The cross-entropy method (CEM) for policy search is a black box optimization (BBO)
    algorithm. This implementation is based on Stulp and Sigaud (2012). Intuitively,
    CEM starts with a multivariate Gaussian dsitribution over policy parameter vectors.
    This distribution has mean thet and covariance matrix Sigma. It then samples some
    fixed number, K, of policy parameter vectors from this distribution. It evaluates
    these K sampled policies by running each one for N episodes and averaging the
    resulting returns. It then picks the K_e best performing policy parameter
    vectors and fits a multivariate Gaussian to these parameter vectors. The mean and
    covariance matrix for this fit are stored in theta and Sigma and this process
    is repeated.

    Parameters
    ----------
    sigma (float): exploration parameter
    theta (numpy.ndarray): initial mean policy parameter vector
    popSize (int): the population size
    numElite (int): the number of elite policies
    numEpisodes (int): the number of episodes to sample per policy
    evaluationFunction (function): evaluates the provided parameterized policy.
        input: theta_p (numpy.ndarray, a parameterized policy), numEpisodes
        output: the estimated return of the policy
    epsilon (float): small numerical stability parameter
    """

    def __init__(self, theta:np.ndarray, sigma:float, popSize:int, numElite:int, numEpisodes:int, evaluationFunction:Callable, epsilon:float=0.0001):

        self._name = "Cross_Entropy_Method"
        
        self._theta = theta #TODO: set this value to the current mean parameter vector
        self._Sigma = sigma*np.eye(self._theta.shape[0]) #TODO: set this value to the current covariance matrix
        self.t = theta
        self.s = self._Sigma
        self.popSize = popSize
        self.numElite = numElite
        self.numEpisodes = numEpisodes
        self.evaluationFunction = evaluationFunction
        self.epsilon = epsilon
        self.bestpolicy = theta
        self.bestJ = -1000000.
        self.eliteJ = None
        # print(self._theta.shape)
        # print(self._theta)
        # print(self._Sigma)


    @property
    def name(self)->str:
        return self._name
    
    @property
    def parameters(self)->np.ndarray:
        # print(self.bestpolicy)
        return self.bestpolicy

    def train(self)->np.ndarray: # need softmax?
        dict = []

        for i in range(self.popSize):

            theta_k = np.random.multivariate_normal(self._theta,self._Sigma)
            # print("theta_k",theta_k)
            J_k = self.evaluationFunction(theta_k,self.numEpisodes)
            # print(J_k)
            dict.append((theta_k,J_k))
            if self.bestJ < J_k:
                self.bestJ = J_k
                self.bestpolicy = theta_k

        dict.sort(key=lambda x: x[1],reverse=True)
        # print(dict)
        Elite_theta = [d[0] for d in dict][:self.numElite]
        self.eliteJ = np.array([d[1] for d in dict][:self.numElite])

        # print(Elite_theta)
        # print("EEEEEEEEEEEEEEE+====================")

        # print(self.bestpolicy)
        self._theta = np.sum(Elite_theta,axis=0)/self.numElite
        # print("dkdkdkdk")
        # print(self._theta)
        tmp = Elite_theta - self._theta
        # print("=================")
        # print(tmp)
        sumTmp = []
        for i in tmp:
            # print("xixi")
            # print(i.shape)
            sumTmp.append(np.dot(i[:,None],i[None,:]))
        Tsum = np.sum(sumTmp,axis = 0)
        # print("===========================================tsum")
        # print(Tsum)
        self._Sigma = (self.epsilon*np.eye(self._theta.shape[0])+Tsum)/(self.epsilon+self.numElite)
        # print(self._theta)
        return self._theta

    def reset(self)->None:
        self._theta = self.t  # TODO: set this value to the current mean parameter vector
        self._Sigma = self.s  # TODO: set this value to the current covariance matrix
        self.bestpolicy = self.t
        self.bestJ = 0.

    def pltJ(self):
        return self.eliteJ.mean(),self.eliteJ.std()