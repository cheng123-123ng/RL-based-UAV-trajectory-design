# 本项目为我的本科毕业论文《基于强化学习的无人机集群控制和安全通信》的对应代码
  本项目是基于RLlib完成的，使用了PPO算法解决了单智能体的问题，QMIX/VDN算法解决了多智能体问题，本项目的主要贡献为建立了一个自己的第三方无人机与无人车通信的环境。有两个环境（分别为单智能体与多智能体环境）都在env目录下面。
  在学习使用RLlib时，发现可以参考的中文资料非常非常少，而大多数英文博客也都写于几年前失去了参考意义，只有比较难懂的官方文档。于是将所遇到并解决的问题都放在代码中供后续研究者使用。
  本文档的英文版为readme_in_english.md
# 本文包括：
  1.如何使用RLLib来训练（train的方式和tune的方式），其中包括了如何保留训练中最优的2个（或n个）模型，并指定他们的位置。
  
  2.如何构建自己的第三方环境，单智能体与多智能体
  
  3.训练完成后，如何重新导入模型。
  
  4.如何自己写trajectory，获得智能体与环境交互的轨迹（PPO和QMIX），据笔者发现，普通算法如PPO的轨迹很容易绘制只需要使用algo.compute_single_action(obs),但是如QMIX这样使用RNN的算法会非常困难，因为网络的输入需要包括之前的状态（这一点RLlib不会帮助你），在查阅QMIX代码后，笔者使用algo.get_policy().compute_actions_from_input_dict(input_dict,timestep=t_step,explore=False)来完成绘制。
  
  5.获得trajectory后如何表示使用plt绘制动态图。
  
  6.使用tensorboard 来观察episode_reward
# 本文结构：
  1.env包括了单智能体与多智能体环境，并在多智能体环境中还有一个是专门用来在render时使用的。
  2.main训练的主体过程，其中的step1包括单智能体的有无窃听者，step3包括多智能体的有无窃听者。
  3.render中有两类代码，trajectory获得agent与env交互的轨迹，和render利用轨迹绘制动态图。
# 使用流程：
##  1.单智能体强化学习：

  
