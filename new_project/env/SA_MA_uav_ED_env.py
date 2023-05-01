from gymnasium.spaces import Dict, Discrete, MultiDiscrete,Box,Tuple
import numpy as np
from gymnasium import Env

from ray.rllib.env.multi_agent_env import MultiAgentEnv, ENV_STATE

import math
import time
from math import exp,log10,pi,atan,log2

import numpy as np
P = 0.1
fc = 2.5 * (10**9)
c = 3 * (10**8)
#suburban,urban,dense urban,high-rise urban
a = [4.88,9.61,12.08,27.23]
b = [0.43,0.16,0.11,0.08]
n_LoS = [0.1,1,1.6,2.3]
n_NLoS = [21,20,23,34]
def com_reward(H, R,env_num):
    # 通讯模型
    mid_a = a[env_num]
    mid_b = b[env_num]
    mid_n_LoS = n_LoS[env_num]
    mid_n_NLoS = n_NLoS[env_num]
    PLS = 0
    PLS += 10*log10(H**2 + R**2)+20*log10(fc) + 20*log10(4*pi/c)+ mid_n_NLoS
    PLS += (mid_n_LoS-mid_n_NLoS)/(1+mid_a*exp(-1*mid_b*(atan(H/R)-mid_a)))

    ##from dB domain transform to real domain
    #PLs =  10**(PLS/10)
    recive = 10*log10(P) - PLS
    ####reward 需要变小
    snr = recive - PLS
    # print("this P:",10*log10(P))
    # print("this N:",PLS)
    # print("this SNR db:",snr)
    snr = 10**(snr/10)
    # print("this SNR :", snr)
    # reward = log2(1+snr)
    ###不能在dB，转化为compesiti



    ###使用距离
    reward = -1/10*math.sqrt(H**2+R**2)
    # print(reward)
    return reward
VIEWPORT_X = 20
VIEWPORT_Y = 20
VIEWPORT_Z = 20
##ucar
start_points_ucar_x = [10,10,7,15,3]
start_points_ucar_y = [10,10,0,0,3]
end_points = [20,20]
##uav
start_points_uav=[10,10,10]
#ED
ED_points = [0,0]

MAX_Ucar_num = 2
MAX_Uav_num = 2
Max_step = 3000

class Uav():
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def state(self):
        return [self.x, self.y, self.z]
    def update(self,action):
        if action==0:
            self.x+=1
        elif action==1:
            self.x-=1
        elif action==2:
            self.y+=1
        elif action==3:
            self.y-=1
        elif action==4:
            self.z+=1
        elif action==5:
            self.z-=1
        return self.state()

class Ucar():
    def __init__(self,start_point_x,start_point_y,end_point_x,end_point_y):
        self.x = start_point_x
        self.y = start_point_y
        self.end_point_x = end_point_x
        self.end_point_y = end_point_y
        self.end = 0
    def state(self):
        return [self.x, self.y,self.end]
    def update(self):
        if self.x != self.end_point_x:
            if self.x < self.end_point_x:
                self.x+=1
            if self.x > self.end_point_x:
                self.x-=1
        elif self.y != self.end_point_y:
            if self.y < self.end_point_y:
                self.y+=1
            if self.y > self.end_point_y:
                self.y-=1
        else :
            self.end = 1
        return self.end
class ED():
    def __init__(self,start_point_x,start_point_y):
        self.x = start_point_x
        self.y = start_point_y
    def state(self):
        return [self.x, self.y]

class MA_UavEnv0(Env):

    def __init__(self,config = None):
 #       super.__init__()

        self.viewer = None  # render viewer
        self.scale = 2      # render scale
        self.frame = 0
        self.n_agent = MAX_Uav_num
        self._agent_ids=set({"uav0","uav1"})
        '''
            0-6
            up,down,left,right,forward,backward,stay
        '''
        self.action_space = Discrete(49)
        '''
            [[x,y,z], Uav
            [x,y,end], Ucar
            ...]
        '''
        # self.observation_space = Dict(
        #     {
        #         "action_mask": Box(0.0, 1.0, shape=(self.action_space.n,),dtype=int),
        #         "observations": Box(low=0, high=VIEWPORT_X, shape=((MAX_Ucar_num * 3 + MAX_Uav_num*3, )),dtype=int),
        #     }
        # )
        #

        self.observation_space = Dict(
            {
                "action_mask": Box(0.0, 1.0, shape=(self.action_space.n,), dtype=int),
                "obs": Box(low=0, high=VIEWPORT_X, shape=((MAX_Ucar_num * 3 + MAX_Uav_num * 3 +2,)), dtype=int),
            }
        )
        self.Uav = None
        self.Ucar = []
        self.ED = None
        self.time_step = 0
        self.ucar_end_num = 0
        self.reset()
    def reset(self, *, seed=None, options=None):
        self.Ucar = []
        self.time_step = 0
        self.ucar_end_num = 0

        self.Uav={i:self.rand_Uav() for i in self._agent_ids}
        self.ED = self.rand_ED()
        self.terminateds = set()
        self.truncateds = set()
        for i in range(MAX_Ucar_num):
            self.Ucar.append(self.rand_Ucar(i))
        a= [self.Uav["uav0"].state()]
        b=[self.Uav["uav1"].state()]
        c=[ucar.state() for (_, ucar) in enumerate(self.Ucar)]
        self.obs = np.concatenate((a,b ,c)).reshape((MAX_Ucar_num * 3 + MAX_Uav_num * 3,))
        self.obs = np.append(self.obs, values=np.array([self.ED.state()]).reshape(2, ), axis=0)

        obs = {"action_mask":self.MA_comput_action_mask(),"observations":self.obs}
        return obs,{}
    def comput_action_mask(self,i):
        action_mask = np.array([1 for i in range(self.action_space.n)])
        if self.Uav[i].state()[0] == VIEWPORT_X:
            action_mask[0] = 0
        if self.Uav[i].state()[0] == 0:
            action_mask[1] = 0
        if self.Uav[i].state()[1] == VIEWPORT_Y:
            action_mask[2] = 0
        if self.Uav[i].state()[1] == 0:
            action_mask[3] = 0
        if self.Uav[i].state()[2] == VIEWPORT_Z:
            action_mask[4] = 0
        if self.Uav[i].state()[2] == 5:
            action_mask[5] = 0
        return action_mask
    def MA_compute_action_mask(self):
        action_mask0 = self.comput_action_mask("uav0")
        action_mask1 = self.comput_action_mask("uav1")
        action_mask = np.array([0 for i in range(49)]).reshape(7,7)
        for i in range(7):
            for j in range(7):
                action_mask[i,j] = 1 if action_mask0[i]==1 or action_mask1[j]==1 else 0
        return action_mask.reshape(49,)
    def step(self, action_dict):
        obs, action_mask,rew, terminated, truncated, info = {}, {}, {}, {}, {}, {}
        reward = 0.0
        action0 = int(action_dict/7)
        action1 = int(math.fmod(action_dict,7))
        self.Uav["uav0"].update(action0)
        self.Uav["uav1"].update(action1)

        flag =0
        self.time_step += 1
        ##upstate ucar_position
        se = {"uav0","uav1"}
        for _, ucar in enumerate(self.Ucar):
            ss = ucar.update()
            re = -10000
            if ss == 0:
                for _,uav_ in enumerate(se):
                    vector1=np.array(self.Uav[uav_].state()[0:2])
                    vector2=np.array(ucar.state()[0:2])
                    H = self.Uav[uav_].state()[2]
                    R = np.linalg.norm(vector1-vector2)
                    mid = com_reward(H, R, 1)
                    re = re if re > mid else mid
                reward +=re
            else:
                flag += 1
        ###reward from ED
        # re = -10000
        # for uav__, _ in action_dict.items():
        #     uav_ = self.Uav[uav__]
        #     vector1 = np.array(uav_.state()[0:2])
        #     vector2 = np.array(self.ED.state())
        #     H = uav_.state()[2]
        #     R = np.linalg.norm(vector1 - vector2)
        #     mid = com_reward(H, R, 1)
        #     re = re if re > mid else mid
        # reward -= re


        self.ucar_end_num = flag
        if self.time_step==30 or self.ucar_end_num == MAX_Ucar_num:
            terminated["__all__"] = True
            truncated["__all__"] = True
        self.obs = np.concatenate(([self.Uav["uav0"].state()], [self.Uav["uav1"].state()],
                                                     [ucar.state() for (_, ucar) in enumerate(self.Ucar)])).reshape(
            (MAX_Ucar_num * 3 + MAX_Uav_num*3,))
        self.obs = np.append(self.obs, values=np.array([self.ED.state()]).reshape(2, ), axis=0)

        return {"action_mask": self.MA_comput_action_mask(), "obs": self.obs}, reward, truncated["__all__"], terminated["__all__"],{}

    def render(self, mode='human'):
        pass

    @staticmethod
    def rand_Ucar(item):
        return Ucar(start_points_ucar_x[item],start_points_ucar_y[item],end_points[0],end_points[1])

    @staticmethod
    def rand_ED():
        return ED(ED_points[0], ED_points[1])

    @staticmethod
    def rand_Uav():
        return Uav(start_points_uav[0],start_points_uav[1] ,start_points_uav[2] )