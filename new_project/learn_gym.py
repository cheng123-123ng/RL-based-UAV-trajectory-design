import gym
import time
import ray
from ray import rllib
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
from ray.rllib.algorithms import ppo
from ray.rllib.examples.models.action_mask_model import (
    ActionMaskModel,
    TorchActionMaskModel,
)
# env = gym.make("UavEnv0-v0")
# env.reset()
# state, reward, done, _ = env.step(7)
# print(state)
# print("reward",reward)
# print("done",done)

config = (
    ppo.PPOConfig()
    # .environment(
    #     # random env with 100 discrete actions and 5x [-1,1] observations
    #     # some actions are declared invalid and lead to errors
    #     ActionMaskEnv,
    #     env_config={
    #         "action_space": Discrete(100),
    #         "observation_space": Box(-1.0, 1.0, (5,)),
    #     },
    # )
    .environment(env="Cartpole-v0")
  #  .training(
        # the ActionMaskModel retrieves the invalid actions and avoids them
   #     model={
    #        "custom_model": ActionMaskModel
    #    },
   # )

)
algo = config.build()
for i in range(10):
     result = algo.train()
     print(pretty_print(result))
     if i % 1 == 0:
         checkpoint_dir = algo.save()
         print(f"Checkpoint saved in directory {checkpoint_dir}")
