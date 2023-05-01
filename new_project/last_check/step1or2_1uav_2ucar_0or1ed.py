import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
from ray.rllib.algorithms import ppo
from ray.rllib.algorithms import a2c
from ray.rllib.algorithms import dqn
from model.action_mask_model import ActionMaskModel
import numpy as np
from env.step1_1uav_1ed import _1Uav_1ED_Env
from plot_picture.render_1uav_1ED import render

if __name__ == "__main__":
    envconfig = {
        "MAX_Ucar_num": 2,
        "MAX_Uav_num": 1,
        "ed_weight": 0,
        "with_state": False,
        "ucar_random": True,
    }
    config = (
        ppo.PPOConfig()
        .environment(_1Uav_1ED_Env)
        .training(
           model={
               "custom_model": ActionMaskModel
           },
        )
    )
    algo = config.build()
    for i in range(50):
        print(i)
        algo.train()
        if i % 10 == 0:
            checkpoint_dir = algo.save()
            print(f"Checkpoint saved in directory {checkpoint_dir}")
    print("-------now start render--------")

    my_env = _1Uav_1ED_Env()
    obs,_ =  my_env.reset()
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
        trajectory["EDx"].append(obs["observations"][9])
        trajectory["EDy"].append(obs["observations"][10])
        trajectory["reward"].append(reward)
        trajectory["episode_reward"].append(episode_reward)
    # Save
    dict = trajectory
    f = f'D:\\桌面\\课程\\毕业设计\\毕业论文\\照片与数据集\\trajectory_1uav_1ED_2ucar_train50.gif'
    render(trajectory, 4, f)

    np.save('D:\\桌面\\课程\\毕业设计\\毕业论文\\new_project\\result\\trajectory_1uav_1ED_2Ucar_train50.npy', dict)  # 注意带上后缀名
    print("---------start plot--------")
    print("ok")
