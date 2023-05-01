
import math
import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from IPython.display import clear_output
import matplotlib.pyplot as plt
use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

user_number = 2
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)
'''def plot(frame_idx, rewards):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.show()
'''
def plot(frame_idx, rewards):
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards))
    plt.plot(rewards)
    plt.show()
class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(ValueNetwork, self).__init__()
        mid = num_inputs[0] * num_inputs[1] + num_actions[0]
        self.linear1 = nn.Linear(mid, hidden_size)
       # self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear2_1 = nn.Linear(hidden_size, hidden_size)
        self.linear2_2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x1 = torch.cat([state[..., 0], state[..., 1]], -1)
        x2 = torch.squeeze(action)
        x = torch.cat([x1, x2], -1)
        #x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear2_1(x))
        x = F.relu(self.linear2_2(x))
        x = self.linear3(x)
        return x


class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(PolicyNetwork, self).__init__()
        mid = num_inputs[0] * num_inputs[1]
        self.linear1 = nn.Linear(mid, hidden_size)
   #     self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear2_1 = nn.Linear(hidden_size, hidden_size)
        self.linear2_2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_actions[0])

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = torch.cat([state[..., 0], state[..., 1]], -1)
        x = x.to(torch.float32)
        x = F.relu(self.linear1(x))
      #  x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear2_1(x))
        x = F.relu(self.linear2_2(x))
        x = F.tanh(self.linear3(x))
   #     x = torch.where(x>0,1,0)
        return x

    def get_action(self, state):
#        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.forward(state)
        action=torch.unsqueeze(action,1)
        #return action
        return action.detach().cpu().numpy()


def ddpg_update(batch_size,
                gamma=0.99,
                min_value=-np.inf,
                max_value=np.inf,
                soft_tau=1e-2):
    state, action, reward, next_state, done = replay_buffer_1.sample(batch_size)

    state = torch.FloatTensor(state).to(device)
    next_state = torch.FloatTensor(next_state).to(device)
    action = torch.FloatTensor(action).to(device)
    reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
    done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

    policy_loss = value_net(state, policy_net(state))
    policy_loss = -policy_loss.mean()

    next_action = target_policy_net(next_state)
    target_value = target_value_net(next_state, next_action.detach())
    expected_value = reward +  gamma * target_value
    expected_value = torch.clamp(expected_value, min_value, max_value)

    value = value_net(state, action)
    value_loss = value_criterion(value, expected_value.detach())

    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()

    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()

    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - soft_tau) + param.data * soft_tau
        )

    for target_param, param in zip(target_policy_net.parameters(), policy_net.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - soft_tau) + param.data * soft_tau
        )
def ddpg_update_2(batch_size,
                gamma=0.99,
                min_value=-np.inf,
                max_value=np.inf,
                soft_tau=1e-2):
    a = 127
    b = 1
    state1, action1, reward1, next_state1, done1= replay_buffer_1.sample(a)
    state, action, reward, next_state, done = replay_buffer_2.sample(b)
    state = torch.concat([torch.FloatTensor(state1),torch.FloatTensor(state)])
    action = torch.concat([torch.FloatTensor(action1),torch.FloatTensor(action)])
    reward = torch.concat([torch.FloatTensor(reward1),torch.FloatTensor(reward)])
    next_state = torch.concat([torch.FloatTensor(next_state1),torch.FloatTensor(next_state)])
    done = torch.concat([torch.FloatTensor(done1),torch.FloatTensor(done)])

    state = torch.FloatTensor(state).to(device)
    next_state = torch.FloatTensor(next_state).to(device)
    action = torch.FloatTensor(action).to(device)
    reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
    done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

    policy_loss = value_net(state, policy_net(state))
    policy_loss = -policy_loss.mean()

    next_action = target_policy_net(next_state)
    target_value = target_value_net(next_state, next_action.detach())
    expected_value = reward +   gamma * target_value
    expected_value = torch.clamp(expected_value, min_value, max_value)

    value = value_net(state, action)
    value_loss = value_criterion(value, expected_value.detach())

    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()

    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()

    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - soft_tau) + param.data * soft_tau
        )

    for target_param, param in zip(target_policy_net.parameters(), policy_net.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - soft_tau) + param.data * soft_tau
        )

class enviroment(object):
    def __init__(self):
        ##对于Hap来说状态包含了当前剩余能量和time_step，一开始能量为100，总time_step为10
        ##对于每一个user来说状态包括当前优先级n（1-2），资源需求R（2-3）
        ##构造11*2的向量，第一个里面为当前能量和时间,对于每一个用户第一个元素为n，第二个为R
        #action 10*1的0，1向量，
        self.num_ueser = user_number
        self.state = torch.zeros([self.num_ueser+1,2],dtype=torch.double)
        self.state[0, 0] = 2*self.num_ueser*10
        self.state[1:, 0] = torch.rand(self.num_ueser) + 1
        self.state[1:, 1] = torch.rand(self.num_ueser) + 2
        self.time_step = 0
        self.end_time_step = 10
        self.done = 0
    def reset(self):
        self.num_ueser = user_number
        self.state = torch.zeros([self.num_ueser+1,2],dtype=torch.double)
        self.state[0, 0] = 2*self.num_ueser*10
        self.state[1:, 0] = torch.rand(self.num_ueser) + 1
        self.state[1:, 1] = torch.rand(self.num_ueser) + 2
        self.time_step = 0
        self.end_time_step = 10
        self.done = 0
        return self.state
    def get_reward(self, action):
        ###同时更改state
        #连接收益函数𝑈(𝑖,𝑡)=𝑎𝑖 𝜂𝑖 log(1+𝑟_𝑖)
        action = torch.tensor(action)
        action = torch.where(action>0,1,0)
        U = action*torch.unsqueeze(self.state[1:, 0],1)*torch.unsqueeze(torch.log(self.state[1:, 1]+1),1)
        #连接成本函数𝑌(𝑖,𝑡)=𝑎𝑖 𝑟(𝑖,𝑡) 𝛽𝑡， 𝛽𝑡=𝛿𝑡/X𝑡 ，𝛿𝑡=1
        Y = action * torch.unsqueeze(self.state[1:, 1] / self.state[0, 0],1)
        #计算回报
        reward = torch.sum(U-Y)

        if self.state[0,0] - torch.sum(torch.tensor(action) * self.state[1:,1]) < 0:
            reward=-1*1000
            self.done=1
        if reward <0 and reward>-900:
            print(reward)
       # if reward >= 6:
       #     print(action)
        #更新HAP状态
        q = self.state[1:,1]
        action = torch.tensor(action)
        action = torch.squeeze(action)
        t = torch.tensor(action) * self.state[1:,1]
        mid =torch.sum(torch.tensor(action) * self.state[1:,1])
        self.state[0,0] = self.state[0,0] - torch.sum(torch.tensor(action) * self.state[1:,1])
        self.state[0,1] = self.state[0,1]+1
        self.time_step = self.state[0,1]
        #更新用户信息
        for i in range(self.num_ueser):
            if action[i] == 1:
                self.state[i+1, 0] = torch.rand(1)+1
                self.state[i+1:, 1] = torch.rand(1)+2
        return reward

    def step(self,action):
        reward = self.get_reward(action)###同时更改state
        if self.time_step == self.end_time_step:
            self.done = 1
        return self.state, reward, self.done
env = enviroment()
state_dim = (user_number+1, 2)
action_dim = (user_number, 1)
hidden_dim = 256

value_net = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)
policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

target_value_net = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)
target_policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
    target_param.data.copy_(param.data)

for target_param, param in zip(target_policy_net.parameters(), policy_net.parameters()):
    target_param.data.copy_(param.data)

value_lr = 1e-3
policy_lr = 1e-4

value_optimizer = optim.Adam(value_net.parameters(), lr=value_lr)
policy_optimizer = optim.Adam(policy_net.parameters(), lr=policy_lr)

value_criterion = nn.MSELoss()

replay_buffer_size = 1000000
replay_buffer_1 = ReplayBuffer(replay_buffer_size)
replay_buffer_2 = ReplayBuffer(replay_buffer_size)
#   max_frames = 12000

'''
    while frame_idx < max_frames:
        state = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            action = policy_net.get_action(state)
            next_state, reward, done = env.step(action)

            replay_buffer.push(state, action, reward, next_state, done)
            if len(replay_buffer) > batch_size:
                ddpg_update(batch_size)

            state = next_state
            episode_reward += reward
            frame_idx += 1

            if frame_idx % max(1000, max_steps + 1) == 0:
                plot(frame_idx, rewards)

            if done:
                break

        rewards.append(episode_reward)'''
def two_agent_com(max_frames=12000):
    max_steps = 10
    frame_idx = 0
    rewards = []
    batch_size = 128
    while frame_idx < max_frames:
        state = env.reset()
        #        ou_noise.reset()
        episode_reward = 0
        for step in range(max_steps):
            action = policy_net.get_action(state.double())
            #            action = ou_noise.get_action(action, step)
            next_state, reward, done = env.step(action)
            if done:
                replay_buffer_2.push(state, action, reward, next_state, done)

            replay_buffer_1.push(state, action, reward, next_state, done)
            # if len(replay_buffer_1) > batch_size:
            #   ddpg_update(batch_size)
            if len(replay_buffer_2) > 30 and len(replay_buffer_1) > 120:
                ddpg_update_2(batch_size)
            state = next_state
            episode_reward += reward
            if done:
                break
     #   if frame_idx % 1000 == 0 and frame_idx > 10:
     #       plot(frame_idx, rewards)
        if frame_idx % 50 == 0 and frame_idx > 10:
            print("step ", frame_idx)
            print("reward", episode_reward)
        frame_idx += 1
        rewards.append(episode_reward)
    return rewards
    #plot(frame_idx, rewards)
