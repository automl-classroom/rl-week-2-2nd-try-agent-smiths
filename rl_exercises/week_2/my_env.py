from __future__ import annotations

import gymnasium as gym
import numpy as np


# ------------- TODO: Implement the following environment -------------
class MyEnv(gym.Env):
    """
    Simple 2-state, 2-action environment with deterministic transitions.

    Actions
    -------
    Discrete(2):
    - 0: move to state 0
    - 1: move to state 1

    Observations
    ------------
    Discrete(2): the current state (0 or 1)

    Reward
    ------
    Equal to the action taken.

    Start/Reset State
    -----------------
    Always starts in state 0.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self):
        """Initializes the observation and action space for the environment."""
        self.observation_space = gym.spaces.Discrete(2)
        self.action_space = gym.spaces.Discrete(2)
        self.state = 0
        self.max_steps = 100
        self.steps = 0

    def reset(self, seed):
        self.state = 0
        self.steps = 0
        return 0, {}

    def step(self, action):
        action = int(action)

        if not self.action_space.contains(action):
            raise RuntimeError("Invalid action taken")

        self.state = action
        self.steps += 1

        return self.state, action, False, self.steps < self.max_steps, {}

    def get_reward_per_action(self):
        return np.array([[0, 1], [0, 1]])

    def get_transition_matrix(self):
        T = np.zeros((2, 2, 2))
        T[0, 0, 0] = 1
        T[0, 1, 1] = 1
        T[1, 0, 0] = 1
        T[1, 1, 1] = 1
        return T


class PartialObsWrapper(gym.Wrapper):
    """Wrapper that makes the underlying env partially observable by injecting
    observation noise: with probability `noise`, the true state is replaced by
    a random (incorrect) observation.

    Parameters
    ----------
    env : gym.Env
        The fully observable base environment.
    noise : float, default=0.1
        Probability in [0,1] of seeing a random wrong observation instead
        of the true one.
    seed : int | None, default=None
        Optional RNG seed for reproducibility.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, env: gym.Env, noise: float = 0.1, seed: int | None = None):
        pass
