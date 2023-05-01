from gymnasium.spaces import Dict, Discrete, MultiDiscrete,Box,Tuple
import numpy as np

from ray.rllib.env.multi_agent_env import MultiAgentEnv, ENV_STATE

import math
import time
from math import exp,log10,pi,atan

import numpy as np
P = 1
fc = 2.5 * (10^9)
c = 3 * (10^8)
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
    PLS += 10*log10(H**2 + R**2)
    PLS += 20*log10(fc) + 20*log10(4*pi/c) + mid_n_NLoS
    PLS += (mid_n_LoS-mid_n_NLoS)/(1+mid_a*exp(-1*mid_b*(atan(H/R)-mid_a)))

    snr = 10*log10(P) - PLS
    return snr
VIEWPORT_X = 20
VIEWPORT_Y = 20
VIEWPORT_Z = 20
##ucar
start_points_ucar_x = [0,0,7,15,3]
start_points_ucar_y = [7,15,0,0,3]

end_points = [20,20]
##uav
start_points_uav=[10,10,10]
AGENT_INIT_SCORE = 5
MAX_Ucar_num = 5
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

class MA_UavEnv0(MultiAgentEnv):

    def __init__(self,config = None):
 #       super.__init__()

        self.viewer = None  # render viewer
        self.scale = 2      # render scale
        self.frame = 0
        self.n_agent = MAX_Uav_num
        self._agent_ids=set({"uav0","uav1"})
        '''
            0-7
            up,down,left,right,forward,backward,stay
        '''
        self.action_space = Discrete(7)
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
                "obs": Box(low=0, high=VIEWPORT_X, shape=((MAX_Ucar_num * 3 + MAX_Uav_num * 3,)), dtype=int),
            }
        )
        self.Uav = None
        self.Ucar = []
        self.time_step = 0
        self.ucar_end_num = 0
        self.reset()
    def reset(self, *, seed=None, options=None):
        self.Ucar = []
        self.time_step = 0
        self.ucar_end_num = 0
        self.Uav={i:self.rand_Uav() for i in self._agent_ids}
        self.terminateds = set()
        self.truncateds = set()
        for i in range(MAX_Ucar_num):
            self.Ucar.append(self.rand_Ucar(i))
        a= [self.Uav["uav0"].state()]
        b=[self.Uav["uav1"].state()]
        c=[ucar.state() for (_, ucar) in enumerate(self.Ucar)]
        self.obs = np.concatenate((a,b ,c)).reshape((MAX_Ucar_num * 3 + MAX_Uav_num * 3,))

        obs = {j:{"action_mask":self.comput_action_mask(j),"obs": self.obs}
               for i, j in enumerate(self._agent_ids)}
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
    def step(self, action_dict):
        obs, action_mask,rew, terminated, truncated, info = {}, {}, {}, {}, {}, {}
        terminated["__all__"]=False
        truncated["__all__"] = False
        reward = 0.0
        for i,action in action_dict.items():
            self.Uav[i].update(action)

        dis = 0
        flag =0
        self.time_step += 1
        ##upstate ucar_position
        for _, ucar in enumerate(self.Ucar):
            ss = ucar.update()
            if ss == 0:
                for uav_,_ in action_dict.items():
                    vector1=np.array(self.Uav[uav_].state()[0:2])
                    vector2=np.array(ucar.state()[0:2])
                    H = self.Uav[uav_].state()[2]
                    R = np.linalg.norm(vector1-vector2)
                    reward += com_reward(H,R,1)
            else:
                flag += 1
        self.ucar_end_num = flag
        if self.time_step==30 or self.ucar_end_num == MAX_Ucar_num:
            terminated["__all__"] = True
            truncated["__all__"] = True
        self.obs = np.concatenate(([self.Uav["uav0"].state()], [self.Uav["uav1"].state()],
                                                     [ucar.state() for (_, ucar) in enumerate(self.Ucar)])).reshape(
            (MAX_Ucar_num * 3 + MAX_Uav_num*3,))
        ##compute action_mask
        # print("step_num",self.time_step)
        # print("action",action_dict)
        # mask0 = self.comput_action_mask("uav0")
        # mask1 = self.comput_action_mask("uav1")
        # print("mask0", mask0)
        # print("mask1", mask1)
        # print("reward",reward)
        # print("observations", self.obs)
        return {j: {"action_mask": self.comput_action_mask(j), "obs": self.obs}
             for i, j in enumerate(self._agent_ids)},{
            "uav0": reward,
            "uav1": reward,
        }, truncated, terminated,{}

    def render(self, mode='human'):
        pass

    @staticmethod
    def rand_Ucar(item):
        return Ucar(start_points_ucar_x[item],start_points_ucar_y[item],end_points[0],end_points[1])

    @staticmethod
    def rand_Uav():
        return Uav(start_points_uav[0],start_points_uav[1] ,start_points_uav[2] )