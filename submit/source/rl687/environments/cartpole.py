import numpy as np
from typing import Tuple
from .skeleton import Environment


class Cartpole(Environment):
    """
    The cart-pole environment as described in the 687 course material. This
    domain is modeled as a pole balancing on a cart. The agent must learn to
    move the cart forwards and backwards to keep the pole from falling.

    Actions: left (0) and right (1)
    Reward: 1 always

    Environment Dynamics: See the work of Florian 2007
    (Correct equations for the dynamics of the cart-pole system) for the
    observation of the correct dynamics.
    """

    def __init__(self):
        self._name = "Cartpole"

        # properly define the variables below
        self._action = None
        self._reward = 1.0  # accumulate
        self._isEnd = False
        self._gamma = 1.0

        # define the state # NOTE: you must use these variable names
        self._x = 0.  # horizontal position of cart
        self._v = 0.  # horizontal velocity of the cart
        self._theta = 0.  # angle of the pole
        self._dtheta = 0.  # angular velocity of the pole

        # dynamics
        self._g = 9.8  # gravitational acceleration (m/s^2)
        self._mp = 0.1  # pole mass
        self._mc = 1.0  # cart mass
        self._l = 0.5  # (1/2) * pole length
        self._dt = 0.02  # timestep
        self._t = 0.0  # total time elapsed  NOTE: you must use this variable
        # print("cartPole")

    @property
    def name(self) -> str:
        return self._name

    @property
    def reward(self) -> float:
        return self._reward

    @property
    def gamma(self) -> float:
        return self._gamma

    @property
    def action(self) -> int:
        return self._action

    @property
    def isEnd(self) -> bool:
        return self._isEnd

    @property
    def state(self) -> np.ndarray:
        state_now = np.array([self._x, self._v, self._theta, self._dtheta])
        # print("state")
        # print(state_now)
        return state_now

    def nextState(self, state: np.ndarray, action: int) -> np.ndarray:
        """
        Compute the next state of the pendulum using the euler approximation to the dynamics
        """
        self._action = action
        # print("State")
        # print(state)
        # print(action)
        _x = state[0]
        _v = state[1]
        _theta = state[2]
        _dtheta = state[3]
        F = -10 if action == 0 else 10
        dwt = (self._g * np.sin(_theta) + np.cos(_theta) * (
                    -F - self._mp * self._l * _dtheta * _dtheta * np.sin(_theta)) / (
                           self._mp + self._mc)) / (self._l * (
                    4.0 / 3.0 - self._mp * np.cos(_theta) * np.cos(_theta) / (self._mp + self._mc)))
        dvt = (F + self._mp * self._l * (
                    _dtheta * _dtheta * np.sin(_theta) - dwt * np.cos(_theta))) / (
                          self._mc + self._mp)
        dxt = _v
        dthetat = _dtheta
        dstate = np.array([dxt, dvt, dthetat, dwt])
        # if state[0]+dstate[0]<20:
        newState = state + self._dt * dstate
        # print("new State")
        # print(newState)
        self._t +=self._dt
        self._reward = self.R(state, action, newState)

        self._x = newState[0]  # horizontal position of cart
        self._v = newState[1]  # horizontal velocity of the cart
        self._theta = newState[2]  # angle of the pole
        self._dtheta = newState[3]  # angular velocity of the pole

        self._isEnd = self.terminal()


        return newState

    def R(self, state: np.ndarray, action: int, nextState: np.ndarray) -> float:
        # print("R")
        if action!=None and np.abs(nextState[2]) > np.pi / 12 or self._t > 20 or np.abs(nextState[0]) > 3:
            return 1.0
        return 1.0

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        takes one step in the environment and returns the next state, reward, and if it is in the terminal state
        """
        # print("step")
        nowState = np.array([self._x, self._v, self._theta, self._dtheta])
        if action == None:
            return nowState, self._reward, self._isEnd
        next_State = self.nextState(nowState, action)

        return next_State, self._reward, self._isEnd

    def reset(self) -> None:
        """
        resets the state of the environment to the initial configuration
        """
        # print("reset")
        self._x = 0.  # horizontal position of cart
        self._v = 0.  # horizontal velocity of the cart
        self._theta = 0.  # angle of the pole
        self._dtheta = 0.  # angular velocity of the pole
        self._t = 0.

        self._action = None
        self._reward = 0.0
        self._isEnd = False


    def terminal(self) -> bool:
        """
        The episode is at an end if:
            time is greater that 20 seconds
            pole falls |theta| > (pi/12.0)
            cart hits the sides |x| >= 3
        """
        # print("terminal")
        if np.abs(self._theta) > np.pi / 12 or self._t > 20 or np.abs(self._x) >= 3:  # end
            return True
        return False
