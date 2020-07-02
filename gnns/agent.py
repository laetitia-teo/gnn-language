"""
This file defines the agent logic and an example game loop.
"""

import torch
import babyai

from torch.nn import Linear

from models import RMC, RMCSparse
from utils import get_graph

env = gym.make('BabyAI-GoToRedBall-v0')

N_ACTIONS = env.action_space.n

class Agent():
    """
    Defines the behavior of the agent.

    b is batch size (predefined);
    N is slot number (predefined and fixed);
    d is embed dim;
    h is number of heads in Self-Attention layer.
    """
    def __init__(self, b, N, d, h):

        self.memory = RMC(N, d, h, b)
        self.actor = Linear(d, N_ACTIONS)
        self.critic = Linear(d, N_ACTIONS)

    def act(self, obs):

        x, batch, ei, _ = get_graph(obs)
        out = self.memory(x, batch, ei)
        action_probas = self.actor(out)
        values = self.critic(out)

        # TODO: sample action probabilistically from policy
        # TODO: entropy/exploration ?
        action = 0

        return action