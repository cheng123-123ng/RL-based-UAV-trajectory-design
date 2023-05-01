from gymnasium.spaces import Tuple, Discrete, Box,Dict
import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.algorithms.qmix import QMixConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.examples.env.two_step_game import TwoStepGame
from model.action_mask_model import ActionMaskModel
import numpy as np

from env.MA_env_test import MA_UavEnv0



def env_creator(args=None):
    env = MA_UavEnv0()
    obs_space = env.observation_space
    act_space = env.action_space
    n_agents = env.n_agent
    obs_space = Tuple([obs_space for _ in range(n_agents)])
    act_space = Tuple([act_space for _ in range(n_agents)])
    grouping = {"group_1": ["uav0","uav1"]}
    return env.with_agent_groups(grouping, obs_space=obs_space, act_space=act_space)

register_env("grouped_test", env_creator)
config = (
    QMixConfig()
    .environment("grouped_test",env_config={
        "MAX_Ucar_num":2,
        "MAX_Uav_num":2,
    })
    .training(mixer='vdn')
    .framework("torch")
)
algo = config.build()
for i in range(2000000):
    print(i)
    algo.train()
    if i % 100 == 0:
        checkpoint_dir = algo.save()
        print(f"Checkpoint saved in directory {checkpoint_dir}")
print("ok")

