import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
from ray.rllib.algorithms import ppo
from ray.rllib.examples.models.action_mask_model import TorchActionMaskModel
from env.env_step1 import _1Uav_1ED_Env
path_to_checkpoint = "C:\\Users\\HW\\ray_results\\test_step1_render\\PPO__1Uav_1ED_Env_b22a3_00000_0_2023-05-07_11-03-09\\checkpoint_000002"
f = f'C:\\Users\\HW\\ray_results\\test_step1_render'#存储轨迹图的地址

envconfig = {
        "MAX_Ucar_num": 2,
        "MAX_Uav_num": 1,
        "ed_num": 1,
        "ed_weight": 1,
        "ucar_random": True,
    }
my_env = _1Uav_1ED_Env(config=envconfig)
obs,_ =  my_env.reset()
config = (
        ppo.PPOConfig()
        .environment(_1Uav_1ED_Env,env_config=envconfig)
        .framework("torch")
        .training(
           model={
               "custom_model": TorchActionMaskModel
           },
        )
    )
algo = config.build()
algo.restore(path_to_checkpoint)

done = False
truncated = False
episode_reward =0
trajectory = {
    "Uavx":[],"Uavy":[],
    "Ucar1x":[],"Ucar1y":[],
    "Ucar2x":[],"Ucar2y":[],
    "EDx": [],"EDy": [],
    "reward":[],"episode_reward":[],
}

while not done and not truncated:
    # action = algo.compute_single_action(obs["observations"])
    action = algo.compute_single_action(obs)
    obs, reward, done, truncated, info = my_env.step(action)
    episode_reward += reward
    # state_queery.append(obs["observations"])
    trajectory["Uavx"].append(obs["observations"][0])
    trajectory["Uavy"].append(obs["observations"][1])
    trajectory["Ucar1x"].append(obs["observations"][3])
    trajectory["Ucar1y"].append(obs["observations"][4])
    trajectory["Ucar2x"].append(obs["observations"][6])
    trajectory["Ucar2y"].append(obs["observations"][7])
    if envconfig["ed_num"] == 1:
        trajectory["EDx"].append(obs["observations"][9])
        trajectory["EDy"].append(obs["observations"][10])
    trajectory["reward"].append(reward)
    trajectory["episode_reward"].append(episode_reward)
# Save
dict = trajectory
from render.step1_render import render
render(trajectory, envconfig["ed_num"], f)

print("---------start plot--------")
print("ok")