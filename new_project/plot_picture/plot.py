import json
# 数据路径
import matplotlib.pyplot as plt
import numpy as np
#QMix_grouped_test_2023-03-28_15-57-04_z27ajqz 5000次 2 5 1
#QMix_grouped_test_2023-03-30_19-48-1812uzyufm 239次修改为capacity后 失败
#QMix_grouped_test_2023-04-03_19-55-55wgqc7qo0 200次 2ucar 成功
#QMix_grouped_test_2023-04-03_20-18-29xc46qx2a 1100  5ucar half
# Opening JSON file
def plot(path=None):
    # if path==None:
    #     path = "C:\\Users\\HW\\ray_results\\QMix_grouped_test_2023-04-11_19-31-460imqhncl\\result.json"
    List = []
    List1 = []
    List2 = []
    # path = "C:\\Users\\HW\\ray_results\\QMix_grouped_test_2023-04-11_19-31-460imqhncl\\result.json"##0-101
    # with open(path) as f:
    #     for jsonObj in f:
    #         Dict = json.loads(jsonObj)
    #         List.append(Dict['info']['learner']['default_policy']["learner_stats"]['loss'])
    #         List2.append(Dict['info']['learner']['default_policy']["learner_stats"]['td_error_abs'])
    #         List1.append(Dict['hist_stats']['episode_reward'])
    # path = "C:\\Users\\HW\\ray_results\\QMix_grouped_test_2023-04-11_19-42-259narxvbe\\result.json"  ##102-150
    # with open(path) as f:
    #     for jsonObj in f:
    #         Dict = json.loads(jsonObj)
    #         List.append(Dict['info']['learner']['default_policy']["learner_stats"]['loss'])
    #         List2.append(Dict['info']['learner']['default_policy']["learner_stats"]['td_error_abs'])
    #         List1.append(Dict['hist_stats']['episode_reward'])
    # path = "C:\\Users\\HW\\ray_results\\QMix_grouped_test_2023-04-11_19-48-56vbzg26op\\result.json"  ##153
    # with open(path) as f:
    #     for jsonObj in f:
    #         Dict = json.loads(jsonObj)
    #         List.append(Dict['info']['learner']['default_policy']["learner_stats"]['loss'])
    #         List2.append(Dict['info']['learner']['default_policy']["learner_stats"]['td_error_abs'])
    #         List1.append(Dict['hist_stats']['episode_reward'])
    # path = "C:\\Users\\HW\\ray_results\\QMix_grouped_test_2023-04-11_19-52-24g33nhpmh\\result.json"  ##153-200
    # with open(path) as f:
    #     for jsonObj in f:
    #         Dict = json.loads(jsonObj)
    #         List.append(Dict['info']['learner']['default_policy']["learner_stats"]['loss'])
    #         List2.append(Dict['info']['learner']['default_policy']["learner_stats"]['td_error_abs'])
    #         List1.append(Dict['hist_stats']['episode_reward'])
    path = "C:\\Users\\HW\\ray_results\\QMix_grouped_test_2023-04-13_13-02-29du2gb92u\\result.json"  ##153-1355
    with open(path) as f:
        for jsonObj in f:
            Dict = json.loads(jsonObj)
            # List.append(Dict['info']['learner']['default_policy']["learner_stats"]['loss'])
            # List2.append(Dict['info']['learner']['default_policy']["learner_stats"]['td_error_abs'])
            List2.append(Dict['sampler_results']['episode_reward_mean'])
            List.append(Dict['sampler_results']['episode_reward_min'])
            List1.append(Dict['sampler_results']['episode_reward_max'])
            # List1.append(Dict['hist_stats']['episode_reward'])
    # x = [i for i in List ]
    # y = np.array([i for i in range(len(x))])
    x = [i for i in List]
    y = np.array([i for i in range(len(x))])

    x2 = [i for i in List2 ]
    y2 = np.array([i for i in range(len(x2))])

    x1 = [i for i in List1]
    y1 = np.array([i for i in range(len(x1))])

    # x1 = [j for i in List1 for j in i]
    # y1 = np.array([i for i in range(len(x1))])

    plt.subplot(1,3,1)
    plt.title("loss")
    plt.plot(y,x)


    plt.subplot(1,3,2)
    plt.title("epsiode_reward")
    plt.plot(y1,x1)

    plt.subplot(1,3,3)
    plt.title("td_error")
    plt.plot(y2,x2)


    plt.show()

    f.close()
if __name__=="__main__":
    path = "C:\\Users\\HW\\ray_results\\QMix_grouped_test_2023-04-11_19-31-460imqhncl\\result.json"
    plot(path)




