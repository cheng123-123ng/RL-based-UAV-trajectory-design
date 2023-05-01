from gymnasium.spaces import Tuple, Discrete, Box,Dict
import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.algorithms.qmix import QMixConfig
from ray.rllib.algorithms.maddpg import MADDPGConfig
from math import modf
import numpy as np

from env.MA_uavenv0 import MA_UavEnv0
from plot_picture.render_2uav_5ucar_0ed import render


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
    .environment("grouped_test")
    .framework("torch")
)
algo = config.build()
for i in range(1):
    print(i)
    algo.train()
    if i % 10 == 0:
        checkpoint_dir = algo.save()
        print(f"Checkpoint saved in directory {checkpoint_dir}")

print("-------now start render--------")
# my_env = MA_UavEnv0()
my_env = env_creator()
obs,_ =  my_env.reset()
done = {}
done["__all__"] = False
truncated = False
episode_reward =0
# trajectory = {
#     "Uav1x":[],"Uav1y":[],
#     "Uav"
#     "Ucar1x":[],"Ucar1y":[],
#     "Ucar2x":[],"Ucar2y":[],
#     "EDx": [],"EDy": [],
#     "reward":[],"episode_reward":[],
# }
#2 uav + 5 ucar
trajectory = [[[] for i in range(2) ] for j in range(7) ]
rew = []
while not done["__all__"]:
    # action = algo.compute_single_action(obs["observations"])
    action = algo.compute_single_action(obs["group_1"])
    obs, reward, done, truncated, info = my_env.step(action)
    # state_queery.append(obs["observations"])
    obs_ = obs["obs"]
    for i in obs_:
        if modf(i,3)==0:
            t=i/3+1
            trajectory[t][0].append(obs_[i])
        if modf(i,3)==1:
            t=i/3+1
            trajectory[t][1].append(obs_[i])
    rew.append(reward["uav0"])
# Save
dict = trajectory
f = f'D:\\桌面\\课程\\毕业设计\\毕业论文\\照片与数据集\\trajectory_1uav_1ED_2ucar_train50.gif'
render(trajectory,rew, f)

np.save('D:\\桌面\\课程\\毕业设计\\毕业论文\\new_project\\result\\trajectory_1uav_1ED_2Ucar_train50.npy', dict)  # 注意带上后缀名
print("---------start plot--------")

print("ok")