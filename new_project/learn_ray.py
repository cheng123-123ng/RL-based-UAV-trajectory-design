import argparse
import os

from gym.spaces import Box, Discrete
import ray
from ray import air, tune
from ray.rllib.algorithms import ppo
from ray.rllib.examples.env.action_mask_env import ActionMaskEnv
from ray.rllib.examples.models.action_mask_model import (
    ActionMaskModel,
    TorchActionMaskModel,
)
from ray.tune.logger import pretty_print
env_config={
                "action_space": Discrete(100),
                "observation_space": Box(-1.0, 1.0, (5,)),
            }
action_space = env_config.get("action_space",Discrete(2))
env = ActionMaskEnv(env_config)
env.reset()
