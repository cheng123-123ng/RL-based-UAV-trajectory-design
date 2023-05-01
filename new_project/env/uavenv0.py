import math
import time

import gymnasium
from gymnasium import spaces
from gymnasium.spaces import Discrete,Box
from gymnasium import Env
from collections import OrderedDict
import numpy as np

VIEWPORT_X = 20
VIEWPORT_Y = 20
VIEWPORT_Z = 10
##ucar
start_points_ucar_x = [0,10]
start_points_ucar_y = [10,0]

end_points = [20,20]
##uav
start_points_uav=[10,10,5]
AGENT_INIT_SCORE = 5
MAX_Ucar_num = 2
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

class UavEnv0(Env):

    def __init__(self,config = None):
        self.viewer = None  # render viewer
        self.scale = 2      # render scale
        self.frame = 0

        '''
            0-7
            up,down,left,right,forward,backward,stay
        '''
        self.action_space = spaces.Discrete(7)
        '''
            [[x,y,z], Uav
            [x,y,end], Ucar
            ...]
        '''
        self.observation_space = spaces.Dict(
            {
                "action_mask": spaces.Box(0.0, 1.0, shape=(self.action_space.n,),dtype=int),
                "observations": spaces.Box(low=0, high=VIEWPORT_X, shape=((MAX_Ucar_num * 3 + 3, )),dtype=int),
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

        self.Uav = self.rand_Uav()

        for i in range(MAX_Ucar_num):
            self.Ucar.append(self.rand_Ucar(i))
        self.state=self.observation_space.sample()
        self.state["observations"] = np.concatenate(([self.Uav.state()], [ucar.state() for (_, ucar) in enumerate(self.Ucar)])).reshape((MAX_Ucar_num * 3 + 3,))
        self.comput_action_mask()
   #     print("step_num",self.time_step)
    #    print("state",self.state)
        return self.state,{}
    def comput_action_mask(self):
        action_mask = np.array([1 for i in range(self.action_space.n)])
        if self.Uav.state()[0] == VIEWPORT_X:
            action_mask[0] = 0
        if self.Uav.state()[0] == 0:
            action_mask[1] = 0
        if self.Uav.state()[1] == VIEWPORT_Y:
            action_mask[2] = 0
        if self.Uav.state()[1] == 0:
            action_mask[3] = 0
        if self.Uav.state()[2] == VIEWPORT_Z:
            action_mask[4] = 0
        if self.Uav.state()[2] == 0:
            action_mask[5] = 0
        self.state["action_mask"] = action_mask
        return action_mask
    def step(self, action: int):
        reward = 0.0
        done = False
        action = action
        dis = 0
        flag =0
        self.time_step += 1
        self.Uav.update(action)
        ##upstate ucar_position
        for _, ucar in enumerate(self.Ucar):
            ss =ucar.update()
            if  ss== 0:
                vector1=np.array(self.Uav.state())
                vector2=np.array([ucar.state()[0],ucar.state()[1],0])
                dis += np.linalg.norm(vector1-vector2)
            else:
                flag += 1
        self.ucar_end_num = flag
        #平均给在路上的供给
        if MAX_Ucar_num-flag !=0 :
            dis =dis / (MAX_Ucar_num-flag)
            reward = 10/(dis+1)

        if self.time_step==30 or self.ucar_end_num == MAX_Ucar_num:
            done=1
        self.state["observations"] = np.concatenate(([self.Uav.state()], [ucar.state() for (_, ucar) in enumerate(self.Ucar)])).reshape((MAX_Ucar_num * 3 + 3,))
        ##compute action_mask
        self.state["action_mask"]=self.comput_action_mask()
        # print("step_num",self.time_step)
        # print("action",action)
        # print("state",self.state)
        return self.state, reward, done, False,{}

    def render(self, mode='human'):
       pass

    @staticmethod
    def rand_Ucar(item):
        return Ucar(start_points_ucar_x[item],start_points_ucar_y[item],end_points[0],end_points[1])

    @staticmethod
    def rand_Uav():
        return Uav(start_points_uav[0],start_points_uav[1] ,start_points_uav[2] )


if __name__ == "__main__":
    my_env = UavEnv0()
    s=my_env.reset()
    print(s)
    my_env.step(1)