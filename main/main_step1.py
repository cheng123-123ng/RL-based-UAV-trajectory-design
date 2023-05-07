import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
from ray.rllib.algorithms import ppo
from ray.rllib.examples.models.action_mask_model import TorchActionMaskModel
from env.env_step1 import _1Uav_1ED_Env
import ray
ray.init(num_cpus=6)
num_rollout_worker = 5
num_envs_per_worker = 2
num_cpus_per_worker = 1
if __name__ == "__main__":
    envconfig = {
        "MAX_Ucar_num": 2,
        "MAX_Uav_num": 1,
        "ed_num": 1,
        "ed_weight": 1,
        "ucar_random": True,
    }
    config = (
        ppo.PPOConfig()
        .environment(_1Uav_1ED_Env,env_config=envconfig)
        .rollouts(num_rollout_workers=num_rollout_worker, num_envs_per_worker=num_envs_per_worker)
        .resources(num_cpus_per_worker=num_cpus_per_worker)
        .framework("torch")
        .training(
           model={
               "custom_model": TorchActionMaskModel
           },
        )
    )
    from ray import tune, air
    from ray.air.config import RunConfig
    stop_iters = 100000
    # stop_timesteps = 1000
    stop_reward = 25
    stop = {
        "training_iteration": stop_iters,
        # "timesteps_total": stop_timesteps,
        # "episode_reward_mean": stop_reward,
    }
    tuner = tune.Tuner(
        "PPO",
        run_config=RunConfig(
            name="test_step1_render",
            local_dir="~/ray_results",
            checkpoint_config=air.CheckpointConfig(
                num_to_keep=2,  # Checkpoints are written to this director
                checkpoint_score_attribute="episode_reward_mean",
                checkpoint_score_order="max",
                checkpoint_frequency=1,  ########这个必须要设置，要不然不能跑，，，
            ),
            stop=stop,
        ),
        param_space=config,
    )
    tuner.fit()
