from gymnasium.spaces import Tuple
from ray.tune.registry import register_env
from ray.rllib.algorithms.qmix import QMixConfig
import numpy as np

from env.MA_env_test import MA_UavEnv0


path_to_checkpoint = "C:\\Users\\HW\\ray_results\\QMix_grouped_test_2023-04-13_18-34-451j5xogy2\\checkpoint_000651"


def env_creator(args):
    env = MA_UavEnv0(args)
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
algo.restore(path_to_checkpoint)

my_env = env_creator({
        "MAX_Ucar_num":2,
        "MAX_Uav_num":2,
    })
obss,_ = my_env.reset()
xx = obss["group_1"]
obs = np.array([j for i in range(2) for value in xx[i].values() for j in value]).reshape(1,42)
input_dict = {
    "obs": np.array([j for i in range(2) for value in xx[i].values() for j in value]).reshape(1,42),
    "prev_actions":np.array([[0,0]]).reshape(1,2),
    "rewards":np.array([0]).reshape(1,),"prev_rewards":np.array([0]).reshape(1,),
    "terminateds":np.array([0]).reshape(1,),"truncateds":np.array([0]).reshape(1,),
    "eps_id": np.array([0]).reshape(1, ), "unroll_id": np.array([0]).reshape(1, ),
    "agent_index": np.array([0]).reshape(1, ), "t": np.array([-1]).reshape(1, ),
    "state_in_0":np.array([0 for i in range(1*2*64)]).reshape(1,2,64),
    "state_out_0":np.array([0 for i in range(1*2*64)]).reshape(1,2,64),
    "seq_lens":np.array([1]).reshape(1,),
}
t_step = 0

trajectory = {
    "Uavx": ([],[] ),"Uavy": ([],[] ),"Uavz": ([],[] ),
    "Ucarx": ([],[]), "Ucary": ([],[],),
    "action":([],[]),
    "EDx": [], "EDy": [],
    "reward": [], "episode_reward": [],
}
episode_reward = 0
while input_dict["terminateds"]==0 and input_dict["truncateds"]==0:
    action,hidden,_ = algo.get_policy().compute_actions_from_input_dict(input_dict,timestep=t_step,explore=False)
    obss,reward, tr, te,_ = my_env.step({
        "uav0":action[0],
        "uav1":action[1],
    })
    obs = []
    obs = obss["group_1"]
    input_dict["obs"] = np.array([j for i in range(2) for value in obs[i].values() for j in value]).reshape(1,42)
    input_dict["prev_actions"] = np.array([[action[0],action[1]]]).reshape(1,2)
    input_dict["prev_rewards"] = input_dict["rewards"]
    input_dict["rewards"] = np.array([reward["group_1"]]).reshape(1,)
    input_dict["terminateds"] = 0 if tr["__all__"]==False else 1
    input_dict["truncateds"] = 0 if te["__all__"]==False else 1
    input_dict["state_in_0"] = hidden[0]
    ####trajectory
    obs = obss["group_1"][0]["obs"]
    trajectory["action"][0].append(action[0])
    trajectory["action"][1].append(action[1])

    trajectory["Uavx"][0].append(obs[0])
    trajectory["Uavy"][0].append(obs[1])
    trajectory["Uavz"][0].append(obs[2])
    trajectory["Uavx"][1].append(obs[3])
    trajectory["Uavy"][1].append(obs[4])
    trajectory["Uavz"][1].append(obs[5])


    trajectory["Ucarx"][0].append(obs[6])
    trajectory["Ucary"][0].append(obs[7])
    trajectory["Ucarx"][1].append(obs[9])
    trajectory["Ucary"][1].append(obs[10])

    trajectory["EDx"].append(obs[12])
    trajectory["EDy"].append(obs[13])
    trajectory["reward"].append(reward["group_1"])
    episode_reward += reward["group_1"]
    trajectory["episode_reward"].append(episode_reward)


# Save
from plot_picture.render_2uav_2ucar_1ed import render
dict = trajectory
f = f'D:\\桌面\\课程\\毕业设计\\毕业论文\\照片与数据集\\trajectory_2uav_2ucar_1ed_train200.gif'
render(trajectory,f)

np.save('D:\\桌面\\课程\\毕业设计\\毕业论文\\new_project\\result\\trajectory_2uav_2Ucar_1ed_train200.npy', dict)  # 注意带上后缀名
print("---------start plot--------")
print("ok")

