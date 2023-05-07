from gymnasium.spaces import Tuple, Discrete, Box,Dict
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'####这两行代码是因为我的gpu比较老，只能下载老的cuda和toch，所以需要加上这两条代码才能保证运行，没有特别的意义
from ray.tune.registry import register_env
from ray.rllib.algorithms.qmix import QMixConfig

from env.env_step3 import MA_UavEnv0
import ray
#设置在训练中最多分配使用的cpu和gpu
ray.init(num_cpus=8,num_gpus=1)
#环境参数设置
envconfig={
    "MAX_Ucar_num":5,
    "MAX_Uav_num":2,
    "ed_num":2,##可以通过调控该参数来设置ed的数量，本代码中最大只能这设置为2，当设置为0时就是第三阶段。
    # 可以通过修改环境中的ed坐标#ED ED_points_x = [10,20]，ED_points_y = [15,10]来扩大ed的数量。
    "ed_weight":3,
    "with_state":False,##使用qmix必须传入代表环境整体的state，state设置难度很大，设置错误会导致效果不如vdn
    "ucar_random":True,##无人车可以每一步随机向终点走，比如当前（0，0）终点为（20，10），下一状态就可能是（1，0）或者（0，1），
    # 如果 "ucar_random":False,就是先动坐标靠近的那个，比如现在x坐标相差为20，y坐标相差为10，那么就先动y，下一状态为（0，1）
}
log_name = "MA_3weight"
##qmix需要经过env.with_agent_groups操作，因为qmix是teamwork形式的多智能体强化学习，全体智能体共享同一个reward。
def env_creator(args):
    env = MA_UavEnv0(args)
    obs_space = env.observation_space
    act_space = env.action_space
    n_agents = env.n_agent
    obs_space = Tuple([obs_space for _ in range(n_agents)])
    act_space = Tuple([act_space for _ in range(n_agents)])
    grouping = {"group_1": ["uav0","uav1"]}
    return env.with_agent_groups(grouping, obs_space=obs_space, act_space=act_space)
register_env("grouped_test", env_creator)#环境需要注册一下，类似于gym中注册为内部环境
##训练参数设置
config = (
    QMixConfig()
    .environment("grouped_test",env_config=envconfig,
    )
    .training(mixer='vdn')
    .framework("torch")
    .rollouts(num_rollout_workers=1,num_envs_per_worker=2)
    .resources(num_cpus_per_worker=7)###因为笔者电脑为8核cpu，经过实验发现每次实验占用的cpu数量为num_rollout_workers*num_cpus_per_worker+1，
                                    #所以只有设置为该参数才能完整使用8核cpu
    # .resources(num_gpus_per_worker=0.9)#若使用gpu可以将gpu数量设置为小数，可能因为笔者电脑很菜，所以还不如不用gpu
    .exploration(
                exploration_config={
                    "final_epsilon": 0.0,
                }
            )
)
from ray import tune,air
from ray.air.config import RunConfig
stop_iters = 100000
# stop_timesteps = 1000
stop_reward = 25
stop = {
        "training_iteration": stop_iters,
         # "timesteps_total": stop_timesteps,
        # "episode_reward_mean": stop_reward,
    }
##开始训练
tuner = tune.Tuner(
    "QMIX",
    run_config=RunConfig(
        name=log_name,#存储地址的名字
        local_dir="~/ray_results",#存储地址
        checkpoint_config=air.CheckpointConfig(
                    num_to_keep = 2,     # Checkpoints are written to this director
                    checkpoint_score_attribute="episode_reward_mean",
                    checkpoint_score_order="max",
                    checkpoint_frequency = 1,########这个必须要设置，要不然不能跑，，，
                ),
        stop=stop,
    ),
    param_space=config,
)
tuner.fit()



