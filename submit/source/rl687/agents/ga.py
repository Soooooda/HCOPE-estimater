import numpy as np
from .bbo_agent import BBOAgent

from typing import Callable


class GA(BBOAgent):
    """
    A canonical Genetic Algorithm (GA) for policy search is a black box 
    optimization (BBO) algorithm. 
    
    Parameters
    ----------
    populationSize (int): the number of individuals per generation
    numEpisodes (int): the number of episodes to sample per policy         
    evaluationFunction (function): evaluates a parameterized policy
        input: a parameterized policy theta, numEpisodes
        output: the estimated return of the policy            
    initPopulationFunction (function): creates the first generation of
                    individuals to be used for the GA
        input: populationSize (int)
        output: a numpy matrix of size (N x M) where N is the number of 
                individuals in the population and M is the number of 
                parameters (size of the parameter vector)
    numElite (int): the number of top individuals from the current generation
                    to be copied (unmodified) to the next generation
    
    """

    def __init__(self, populationSize:int, evaluationFunction:Callable, 
                 initPopulationFunction:Callable, numElite:int=1, numEpisodes:int=10,numParents:int=5,arpha:float=1.0):
        self._name = "Genetic_Algorithm"

        self.populationSize = populationSize
        self.evalutationFunction = evaluationFunction
        self.initPopulationFunction = initPopulationFunction
        self.numElite = numElite
        self.numEpisodes = numEpisodes
        self._population = self.initPopulationFunction(self.populationSize)  # TODO: set this value to the most recently created generation
        self.numParents = numParents
        self.arpha = arpha
        self.bestpolicy = self._population[0]
        self.bestJ = self.evalutationFunction(self._population[0], self.numEpisodes)
        self.eliteJ = None
        print(numElite)

    @property
    def name(self)->str:
        return self._name
    
    @property
    def parameters(self)->np.ndarray:
        # print("para")
        # print(self.bestpolicy)
        return np.array(self.bestpolicy)

    def _mutate(self, parent:np.ndarray)->np.ndarray:

        """
        Perform a mutation operation to create a child for the next generation.
        The parent must remain unmodified. 
        
        output:
            child -- a mutated copy of the parent
        """
        # print("mutate")
        child = parent + self.arpha * np.random.normal(0, 1,parent.shape[0])
        return child


    def train(self)->np.ndarray:
        # print("train")
        dict = []
        # print(self._population)
        for k in range(self.populationSize):
            J_k = self.evalutationFunction(self._population[k], self.numEpisodes)
            # print(self._population[k])
            dict.append((self._population[k], J_k))
            if self.bestJ<J_k:
                self.bestJ = J_k
                self.bestpolicy = self._population[k]
        dict.sort(key=lambda x: x[1], reverse=True)
        Elite_theta = np.array([d[0] for d in dict][:self.numElite])
        self.eliteJ = np.array([d[1] for d in dict][:self.numElite])
        all_theta = np.array([d[0] for d in dict])
        # self.bestpolicy = np.array(Elite_theta[0])
        # print("elite")
        # print(Elite_theta)
        # print(self.bestpolicy)
        parents = all_theta[:self.numParents]  # pop 100 parents 20 elite 30 so generate 70 children
        child_num = self.populationSize - self.numElite
        # get children
        count = 0
        children = []
        for i in range(child_num):
            crossover = (parents[count]+parents[(count + self.numParents + 1) % self.numParents])/2
            child = self._mutate(parents[count])
            count = (count + self.numParents + 1) % self.numParents
            children.append((child))
        self._population = np.concatenate((Elite_theta,np.array(children)),axis=0)
        # print("epoch___")
        # print(self._population)
        return self.bestpolicy#,self._population)

    def reset(self)->None:
        print("reset")
        self._population = self.initPopulationFunction(self.populationSize)  # TODO: set this value to the most recently created generation
        self.bestpolicy = None
        self.bestJ = 0.

    def pltJ(self):
        return self.eliteJ.mean(),self.eliteJ.std()