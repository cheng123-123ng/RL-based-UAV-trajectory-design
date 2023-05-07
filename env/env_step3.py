from gymnasium.spaces import Dict, Discrete, MultiDiscrete,Box,Tuple
import numpy as np

from ray.rllib.env.multi_agent_env import MultiAgentEnv, ENV_STATE

import math
import time
from math import exp,log10,pi,atan,log2

P = 0.4
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
    PLS += 10*log10(H**2 + R**2)+20*log10(fc) + 20*log10(4*pi/c)#+ mid_n_NLoS
    PLS += (mid_n_LoS-mid_n_NLoS)/(1+mid_a*exp(-1*mid_b*(atan(H/R)-mid_a)))

    Pdb = 10*log10(P)
    re = Pdb - PLS
    noise = -50
    snr = re - noise
    snr = 10 ** (snr / 10)
    ca = log2(1 + snr)
    reward = ca


    # ###使用距离
    # reward = -0.1*math.sqrt(H**2+R**2)
    # print(reward)
    return reward
VIEWPORT_X = 25  ## 1000
VIEWPORT_Y = 25  ## 1000
VIEWPORT_Z = 20 ##70-120
##ucar
start_points_ucar_x = [3,7,5,15,8]
start_points_ucar_y = [7,3,15,5,8]
end_points = [VIEWPORT_X,VIEWPORT_Y]
##uav
start_points_uav_x = [1,5]
start_points_uav_y = [5,3]
start_points_uav_z = [10,10]
#ED
ED_points_x = [10,20]
ED_points_y = [15,10]


Max_step = 3000000

class Uav():
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.v = 2

    def state(self):
        return [self.x, self.y, self.z]
    def update(self,action):
        if action==0:
            self.x+=self.v
        elif action==1:
            self.x-=self.v
        elif action==2:
            self.y+=self.v
        elif action==3:
            self.y-=self.v
        elif action==4:
            self.z+=self.v
        elif action==5:
            self.z-=self.v
        return self.state()

class Ucar():
    def __init__(self,start_point_x,start_point_y,end_point_x,end_point_y,random_car):
        self.x = start_point_x
        self.y = start_point_y
        self.end_point_x = end_point_x
        self.end_point_y = end_point_y
        self.end = 0
        self.random_car = random_car
    def state(self):
        return [self.x, self.y,self.end]
    def update(self):
        from math import fabs
        dx = fabs(self.end_point_x - self.x)
        dy = fabs(self.end_point_y - self.y)
        if self.random_car == False:
            if dx<=dy:
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
                else:
                    self.end = 1
            else:
                if self.y != self.end_point_y:
                    if self.y < self.end_point_y:
                        self.y+=1
                    if self.y > self.end_point_y:
                        self.y-=1
                elif self.x != self.end_point_x:
                    if self.x < self.end_point_x:
                        self.x+=1
                    if self.x > self.end_point_x:
                        self.x-=1
                else:
                    self.end = 1
            return self.end
        else:
            from random import randint
            if dx>0 and dy>0:
                flag = randint(0,1)
                if flag == 0:
                    self.x+=1
                else:
                    self.y+=1
            elif dx>0 and dy==0:
                self.x+=1
            elif dx==0 and dy>0:
                self.y+=1
            else:
                self.end = 1
            return self.end
class ED():
    def __init__(self,start_point_x,start_point_y):
        self.x = start_point_x
        self.y = start_point_y
    def state(self):
        return [self.x, self.y]

class MA_UavEnv0(MultiAgentEnv):

    def __init__(self,config = None):
 #       super.__init__()
        self.MAX_Ucar_num = 5 if config==None else config.get("MAX_Ucar_num")
        self.MAX_Uav_num = 2 if config==None else config.get("MAX_Uav_num")
        self.ed_num = 2 if config==None else config.get("ed_num")
        self.ed_weight = 0 if config==None else config.get("ed_weight")
        self.nomal_obs = False if config==None else config.get("normal_obs")
        self.with_state = True if config == None else config.get("with_state")
        self.viewer = None  # render viewer
        self.scale = 2      # render scale
        self.frame = 0
        self.n_agent = self.MAX_Uav_num
        self.state = None
        self._agent_ids=set({"uav0","uav1"})
        self.ucar_random = False if config == None else config.get("ucar_random")
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
        self.max_dis = math.sqrt(25**2+25**2+10**2)
        self.max_re_x = 25
        self.max_re_y = 25
        if self.with_state == False:
            self.observation_space = Dict(
                {
                    "action_mask": Box(0.0, 1.0, shape=(self.action_space.n,), dtype=int),
                    "obs": Box(low=-1*self.max_dis, high=self.max_dis, shape=(((self.MAX_Uav_num-1)* 4 + self.MAX_Ucar_num * 4 +self.ed_num*4,)), dtype=int),
                }
            )
        else:
            self.observation_space = Dict(
                {
                    "action_mask": Box(0.0, 1.0, shape=(self.action_space.n,), dtype=int),
                    "obs": Box(low=-1 * self.max_dis, high=self.max_dis,
                               shape=(((self.MAX_Uav_num - 1) * 4 + self.MAX_Ucar_num * 4 + self.ed_num * 4,)), dtype=int),
                    ENV_STATE:Box(low=-5, high=VIEWPORT_X, shape=((self.MAX_Uav_num* 3 + self.MAX_Ucar_num * 3 +self.ed_num*2,)), dtype=int),
                }
            )
        self.Uav = None
        self.Ucar = []
        self.ED = []
        self.time_step = 0
        self.ucar_end_num = 0
        self.reset()
    def reset(self, *, seed=None, options=None):
        self.Ucar = []
        self.obs = [[],[]]
        self.ED = []
        self.time_step = 0
        self.ucar_end_num = 0
        _agent_ids = {"uav0", "uav1"}
        self.Uav={j:self.rand_Uav(i) for i,j in enumerate( _agent_ids)}
        for i in range(self.ed_num):
            self.ED.append(self.rand_ED(i))
        self.terminateds = set()
        self.truncateds = set()
        for i in range(self.MAX_Ucar_num):
            self.Ucar.append(self.rand_Ucar(i,self.ucar_random))
        for i in range(self.MAX_Uav_num):
            self.get_obs(i)
        if self.with_state == True:
            obs = {j:{"action_mask":self.comput_action_mask(j),
                      "obs": self.obs[i],
                      ENV_STATE:self.state,
                      }
                   for i, j in enumerate(self._agent_ids)}
        else:
            obs = {j:{"action_mask":self.comput_action_mask(j),
                      "obs": self.obs[i],
                      }
                   for i, j in enumerate(self._agent_ids)}
        return obs,{}
    def get_obs(self,i):
        ##类型，1-uav,0-uac,-1-ed
        obs = np.zeros((self.MAX_Uav_num-1+self.MAX_Ucar_num+self.ed_num,4))
        id_list = ["uav0","uav1"]
        a = self.Uav[id_list[i]].state()
        ######## other uav
        id_list.pop(i)
        for t,j in enumerate(id_list):
            b = self.Uav[j].state()
            re_x0 = math.fabs(b[0] - a[0])
            re_y0 = math.fabs(b[1] - a[1])
            dis0 = math.sqrt(re_x0 ** 2 + re_y0 ** 2)
            obs[t]=np.array([re_x0,re_y0,dis0,0])
        #####ucar
        for (j, ucar) in enumerate(self.Ucar):
            ucar = ucar.state()
            re_x1 = math.fabs(ucar[0]-a[0])
            re_y1 = math.fabs(ucar[1]-a[1])
            dis1 = math.sqrt(re_x1**2+re_y1**2)
            obs[self.MAX_Uav_num-1+j] = np.array([re_x1, re_y1, dis1, 1])
        for (j, ed) in enumerate(self.ED):
            ed = ed.state()
            re_x2 = math.fabs(ed[0] - a[0])
            re_y2 = math.fabs(ed[1] - a[1])
            dis2 = math.sqrt(re_x2 ** 2 + re_y2 ** 2)
            obs[self.MAX_Uav_num+self.MAX_Ucar_num-1+j] = np.array([re_x2, re_y2, dis2, -1])
        self.obs[i] = obs.reshape(((self.MAX_Uav_num-1)* 4 + self.MAX_Ucar_num * 4 +self.ed_num*4,))
        ##############get ENV_state
        a= [self.Uav["uav0"].state()]
        b=[self.Uav["uav1"].state()]
        c=[ucar.state() for (_, ucar) in enumerate(self.Ucar)]
        d=[ed.state() for (_, ed) in enumerate(self.ED)]
        mid= np.concatenate((a, b, c)).reshape((self.MAX_Ucar_num * 3 + self.MAX_Uav_num * 3,))
        self.state = np.append(mid, values=np.array(d).reshape(self.ed_num*2, ), axis=0)

        # self.obs[0] = self.obs[0] / arr
        # self.obs[1] = self.obs[1] / arr
    def comput_action_mask(self,i):
        action_mask = np.array([1 for i in range(self.action_space.n)])
        if self.Uav[i].state()[0] >= VIEWPORT_X:
            action_mask[0] = 0
        if self.Uav[i].state()[0] <= 0:
            action_mask[1] = 0
        if self.Uav[i].state()[1] >= VIEWPORT_Y:
            action_mask[2] = 0
        if self.Uav[i].state()[1] <= 0:
            action_mask[3] = 0
        if self.Uav[i].state()[2] >= VIEWPORT_Z:
            action_mask[4] = 0
        if self.Uav[i].state()[2] <= 5:
            action_mask[5] = 0
        # action_mask = action_mask * 0
        # action_mask[0] = 1
        action_mask[4] = 0
        action_mask[5] = 0
        return action_mask
    def step(self, action_dict):
        obs, action_mask,rew, terminated, truncated, info = {}, {}, {}, {}, {}, {}
        self.obs = [[], []]
        terminated["__all__"]=False
        truncated["__all__"] = False
        reward = 0.0
        for i ,action in action_dict.items():
            self.Uav[i].update(action)

        dis = 0
        flag =0
        self.time_step += 1
        ##upstate ucar_position
        for _, ucar in enumerate(self.Ucar):
            ss = ucar.update()
            re = -10000
            if ss == 0:
                for uav_,_ in action_dict.items():
                    vector1=np.array(self.Uav[uav_].state()[0:2])
                    vector2=np.array(ucar.state()[0:2])
                    H = self.Uav[uav_].state()[2]
                    R = np.linalg.norm(vector1-vector2)
                    mid = com_reward(H, R, 1)
                    re = re if re > mid else mid
                reward += re
            else:
                flag += 1
        ED = True
        if ED == True:
            ##reward from ED
            for _, ed in enumerate(self.ED):
                re = 0
                for uav__, _ in action_dict.items():
                    uav_ = self.Uav[uav__]
                    vector1 = np.array(uav_.state()[0:2])
                    vector2 = np.array(ed.state())
                    H = uav_.state()[2]
                    R = np.linalg.norm(vector1 - vector2)
                    mid = com_reward(H, R, 1)
                    if R <=2 :
                        re = re if re > mid else mid
                reward -= self.ed_weight*re


        self.ucar_end_num = flag
        if self.time_step==Max_step or self.ucar_end_num == self.MAX_Ucar_num:
            terminated["__all__"] = True
            truncated["__all__"] = True
        for i in range(self.MAX_Uav_num):
            self.get_obs(i)
        if self.with_state == True:
            return {j: {"action_mask": self.comput_action_mask(j),
                        "obs": self.obs[i],
                        ENV_STATE: self.state,
                        }
                    for i, j in enumerate(self._agent_ids)}, {
                       "uav0": reward,
                       "uav1": reward,
                   }, truncated, terminated, {}
        else:
            return {j: {"action_mask": self.comput_action_mask(j),
                        "obs": self.obs[i],
                        }
                 for i, j in enumerate(self._agent_ids)},{
                "uav0": reward,
                "uav1": reward,
            }, truncated, terminated,{}
    def render(self, mode='human'):
        pass

    @staticmethod
    def rand_Ucar(item,ucar_random):
        return Ucar(start_points_ucar_x[item],start_points_ucar_y[item],end_points[0],end_points[1],ucar_random)
    @staticmethod
    def rand_ED(i):
        return ED(ED_points_x[i], ED_points_y[i])

    @staticmethod
    def rand_Uav(item):
        return Uav(start_points_uav_x[item],start_points_uav_y[item] ,start_points_uav_z[item] )
if __name__=="__main__":
    my_env = MA_UavEnv0()
    my_env.reset()