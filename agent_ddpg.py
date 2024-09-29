import random
import numpy as np
import torch
import torchkeras
import torch.nn as nn
import torch.optim as optim

from collections import deque
# 选择设备，使用gpu还是cpu进行计算
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device type:", device)

# 超参数
LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
GAMMA = 0.99
MEMORY_SIZE = 10000
BATCH_SIZE = 64
TAU = 5e-3

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_dim)# 使用全连接层

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        # print(f"Actor layer 1 output shape: {x.shape}")
        x = torch.relu(self.fc2(x))
        # print(f"Actor layer 2 output shape: {x.shape}")
        action = torch.tanh(self.fc3(x))
        # print(f"Actor layer 3 output shape: {action.shape}")
        return action

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        # print(f"Critic input shape: {x.shape}")
        x = torch.relu(self.fc1(x))
        # print(f"Critic layer 1 output shape: {x.shape}")
        x = torch.relu(self.fc2(x))
        # print(f"Critic layer 2 output shape: {x.shape}")
        output = self.fc3(x)
        # print(f"Critic layer 3 output shape: {output.shape}")
        return output
class ReplayMemory:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)# deque队列来进行记忆的添加

    def add_memo(self, state, action, reward, next_state, done):# 存经验的方法，对应上main方法中的add_memo(state,action,reward,next_state,done)
        # state = np.expand_dims(state, 0)# 进行升维度,将状态的维度从一列三行（3,1） 升维度到一行三列（1,3）
        # next_state = np.expand_dims(next_state, 0)# next_state同样也升维度
        self.buffer.append((state, action, reward, next_state, done))# append是指从buffer的右端插入，以此增加经验

    def sample(self, batch_size):  # 取出经验，取出batch_size个经验
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    def __len__(self):
        return len(self.buffer)
class OUNoise:
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state
class DDPGAgent:
    def __init__(self, state_dim, action_dim):#DDPG agent是由critic，actor和memo三个构成的
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        self.replay_buffer = ReplayMemory(MEMORY_SIZE)# 初始化
        self.ou_noise = OUNoise(action_dim)# 实例化OUNoise


    def get_action(self, state, noise_scale=0.1):
        state = torch.FloatTensor(state).to(device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy()
        self.actor.train()
        noise = self.ou_noise.noise() * noise_scale
        action = action + noise
        return np.clip(action, 0, 1)

    def update(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(BATCH_SIZE)
        states = torch.FloatTensor(states).view(BATCH_SIZE, -1)[:, :480].to(device)# 确保是（64,800）
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).view(BATCH_SIZE, -1)[:, :480].to(device)# 确保是（64,800）
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)  # 将它们都转成tensor数据类型并使用gpu进行加速


        # 更新critic网络
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_Q = self.critic_target(next_states, next_actions)
            target_Q = rewards + (GAMMA*target_Q*(1-dones))
        current_Q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_Q, target_Q)
        self.critic_optimizer.zero_grad()# 清除之前的梯度信息
        critic_loss.backward()# 计算loss的梯度
        self.critic_optimizer.step()#更新critic的参数

        # 更新actor网络，用策略梯度的方式来更新actor网络
        actor_loss = -torch.mean(self.critic(states, self.actor(states)))#-Q的期望求平均
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        #更新target actor和target critic参数
        for target_param, param, in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        for target_param, param, in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
