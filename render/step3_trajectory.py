import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
from gymnasium.spaces import Tuple
from ray.tune.registry import register_env
from ray.rllib.algorithms.qmix import QMixConfig
import numpy as np

import env.MA_env_test as en
from env.env_step3 import MA_UavEnv0

path_to_checkpoint = "C:\\Users\\HW\\ray_results\\test_render\\QMIX_grouped_test_32753_00000_0_2023-05-02_15-51-50\checkpoint_000029"
envconfig={
    "MAX_Ucar_num":5,
    "MAX_Uav_num":2,
    "ed_num":0,##可以通过调控该参数来设置ed的数量，本代码中最大只能这设置为2，当设置为0时就是第三阶段。
    # 可以通过修改环境中的ed坐标#ED ED_points_x = [10,20]，ED_points_y = [15,10]来扩大ed的数量。
    "ed_weight":10,
    "with_state":False,##使用qmix必须传入代表环境整体的state，state设置难度很大，设置错误会导致效果不如vdn
    "ucar_random":True,##无人车可以每一步随机向终点走，比如当前（0，0）终点为（20，10），下一状态就可能是（1，0）或者（0，1）
    "test":"safe",###can be set as "safe","normal","communication"
}

def env_creator(args):
    env = MA_UavEnv0(args)
    obs_space = env.observation_space
    act_space = env.action_space
    n_agents = env.n_agent
    obs_space = Tuple([obs_space for _ in range(n_agents)])
    act_space = Tuple([act_space for _ in range(n_agents)])
    grouping = {"group_1": ["uav0","uav1"]}
    return env.with_agent_groups(grouping, obs_space=obs_space, act_space=act_space)
def my_env_creator(args):
    env = en.MA_UavEnv0(args)
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
    .environment("grouped_test",env_config=envconfig,)
    .framework("torch")
    .training(mixer='vdn')
)
algo = config.build()
algo.restore(path_to_checkpoint)

my_env = env_creator(envconfig)
##my_env中的obs保持为各个object的坐标
obss,_ = my_env.reset()
my_env_test = my_env_creator(envconfig)
obs_test = my_env_test.reset()
if envconfig["with_state"]:
    # state_size = 64
    xx =[{
        "action_mask":obss["group_1"][0]["action_mask"],
        "obs":obss["group_1"][0]["obs"],
        "state":obss["group_1"][0]["state"],
          },
        {
            "action_mask": obss["group_1"][1]["action_mask"],
            "obs": obss["group_1"][1]["obs"],
            "state": obss["group_1"][1]["state"],
        }
    ]
    obs_size = (envconfig["MAX_Uav_num"]-1 + envconfig["MAX_Ucar_num"] + envconfig["ed_num"]) * 4\
               + 7 \
               + envconfig["MAX_Uav_num"]*3 + envconfig["MAX_Ucar_num"]*3 +envconfig["ed_num"]*2
else:
    xx =[{
        "action_mask":obss["group_1"][0]["action_mask"],
        "obs":obss["group_1"][0]["obs"],
    },
        {
            "action_mask": obss["group_1"][1]["action_mask"],
            "obs": obss["group_1"][1]["obs"],
        }
    ]
    obs_size = (envconfig["MAX_Uav_num"]-1 + envconfig["MAX_Ucar_num"] + envconfig["ed_num"]) * 4 + 7
obs = np.array([j for i in range(2) for value in xx[i].values() for j in value]).reshape(1,obs_size*2)
input_dict = {
    "obs": np.array([j for i in range(2) for value in xx[i].values() for j in value]).reshape(1,obs_size*2),
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
    "Ucarx": ([],[],[],[],[]), "Ucary": ([],[],[],[],[]),
    "action":([],[]),
    "EDx": ([],[] ), "EDy":([],[] ),
    "reward": [], "episode_reward": [],
}
episode_reward = 0
while input_dict["terminateds"]==0 and input_dict["truncateds"]==0:
    ##因为QMIX系列算法使用的是rnn，不能简单的使用compute_single_action
    action,hidden,_ = algo.get_policy().compute_actions_from_input_dict(input_dict,timestep=t_step,explore=False)
    obss,reward, tr, te,_ = my_env.step({
        "uav0":action[0],
        "uav1":action[1],
    })
    obs = []
    obs = obss["group_1"]
    input_dict["obs"] = np.array([j for i in range(2) for value in obs[i].values() for j in value]).reshape(1,obs_size*2)
    input_dict["prev_actions"] = np.array([[action[0],action[1]]]).reshape(1,2)
    input_dict["prev_rewards"] = input_dict["rewards"]
    input_dict["rewards"] = np.array([reward["group_1"]]).reshape(1,)
    input_dict["terminateds"] = 0 if tr["__all__"]==False else 1
    input_dict["truncateds"] = 0 if te["__all__"]==False else 1
    input_dict["state_in_0"] = hidden[0]
    ####trajectory
    obss_test, reward_test, _, _, _ = my_env_test.step({
        "uav0": action[0],
        "uav1": action[1],
    })
    obs = obss_test["group_1"][0]["obs"]
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
    trajectory["Ucarx"][2].append(obs[12])
    trajectory["Ucary"][2].append(obs[13])
    trajectory["Ucarx"][3].append(obs[15])
    trajectory["Ucary"][3].append(obs[16])
    trajectory["Ucarx"][4].append(obs[18])
    trajectory["Ucary"][4].append(obs[19])
    for i in range(envconfig["ed_num"]):
        trajectory["EDx"][i].append(obs[21+2*i])
        trajectory["EDy"][i].append(obs[22+2*i])
        # trajectory["EDx"][1].append(obs[23])
        # trajectory["EDy"][1].append(obs[24])
    trajectory["reward"].append(reward_test["group_1"])
    episode_reward += reward_test["group_1"]
    trajectory["episode_reward"].append(episode_reward)

print(episode_reward)
# Save
from step3_render import render
dict = trajectory
f = f'D:\\桌面\\课程\\毕业设计\\毕业论文\\照片与数据集\\trajectory_2uav_2ucar_1ed_train200.gif'
render(trajectory,envconfig["ed_num"],f)

np.save('D:\\桌面\\课程\\毕业设计\\毕业论文\\new_project\\result\\trajectory_2uav_2Ucar_1ed_train200.npy', dict)  # 注意带上后缀名
print("---------start plot--------")
print("ok")



