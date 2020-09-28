#!/usr/bin/env python3

"""
Script to train the agent through reinforcment learning.
"""

import os
import logging
import csv
import json
import gym
import time
import datetime
import torch
import numpy as np
import subprocess

import babyai
import babyai.utils as utils
import babyai.rl
from babyai.arguments import ArgumentParser
from babyai.model_gnn import ACModelGNN
from babyai.evaluate import batch_evaluate
from babyai.utils.agent import RandomAgent

# Parse arguments
parser = ArgumentParser()
parser.add_argument("--algo", default='ppo',
                    help="algorithm to use (default: ppo)")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--reward-scale", type=float, default=20.,
                    help="Reward scale multiplier")
parser.add_argument("--gae-lambda", type=float, default=0.99,
                    help="lambda coefficient in GAE formula (default: 0.99, 1 means no gae)")
parser.add_argument("--value-loss-coef", type=float, default=0.5,
                    help="value loss term coefficient (default: 0.5)")
parser.add_argument("--max-grad-norm", type=float, default=0.5,
                    help="maximum norm of gradient (default: 0.5)")
parser.add_argument("--clip-eps", type=float, default=0.2,
                    help="clipping epsilon for PPO (default: 0.2)")
parser.add_argument("--ppo-epochs", type=int, default=4,
                    help="number of epochs for PPO (default: 4)")
parser.add_argument("--save-interval", type=int, default=50,
                    help="number of updates between two saves (default: 50, 0 means no saving)")

args = parser.parse_args()
args.env = 'BabyAI-GoToRedBall-v0'
args.procs = 1
# args.frames_per_proc = 40
args.memory_dim = (4, 128)
args.image_dim = 5
args.arch = 'gnn'

args.seed = 41
utils.seed(args.seed)

# Testing the model before saving
test_env_name = args.env
agent = RandomAgent(seed=args.seed)
logs = batch_evaluate(agent, test_env_name, args.val_seed, args.val_episodes)
mean_return = np.mean(logs["return_per_episode"])
success_rate = np.mean([1 if r > 0 else 0 for r in logs['return_per_episode']])

print(mean_return)
print(success_rate)

