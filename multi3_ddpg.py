import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import sys
sys.stdout = sys.__stdout__
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device type:", device)

# Hyperparameters
LR_ACTOR = 1e-6
LR_CRITIC = 1e-5
GAMMA = 0.99
MEMORY_SIZE = 1000000
BATCH_SIZE = 256
TAU = 5e-3
num_agents = 3
class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim, num_agents):
        self.capacity = capacity
        self.buffer = {
            "states": np.zeros((capacity, num_agents, state_dim // num_agents)),
            "actions": np.zeros((capacity, num_agents, action_dim)),
            "rewards": np.zeros((capacity, num_agents)),
            "next_states": np.zeros((capacity, num_agents, state_dim // num_agents)),
            "dones": np.zeros((capacity, num_agents))
        }
        self.current = 0
        self.size = 0

    def add_memo(self, states, actions, rewards, next_states, dones):
        idx = self.current % self.capacity
        self.buffer["states"][idx] = states
        self.buffer["actions"][idx] = actions.reshape(num_agents, -1)
        self.buffer["rewards"][idx] = rewards
        self.buffer["next_states"][idx] = next_states
        self.buffer["dones"][idx] = dones

        self.current += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxes = np.random.choice(self.size, batch_size, replace=False)
        batch = {k: v[idxes] for k, v in self.buffer.items()}
        return batch

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x))
        return action

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, num_agents):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim*num_agents, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)
        return output

class OUNoise:
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.1):
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

class MADDPGAgent:
    def __init__(self, state_dim, action_dim, num_agents):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.update_counter = 0
        self.actors = [Actor(state_dim // num_agents, action_dim).to(device) for _ in range(num_agents)]
        self.actor_targets = [Actor(state_dim // num_agents, action_dim).to(device) for _ in range(num_agents)]
        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=LR_ACTOR) for actor in self.actors]

        self.critics = [Critic(state_dim, action_dim, num_agents).to(device) for _ in range(num_agents)]
        self.critic_targets = [Critic(state_dim, action_dim, num_agents).to(device) for _ in range(num_agents)]
        self.critic_optimizers = [optim.Adam(critic.parameters(), lr=LR_CRITIC) for critic in self.critics]

        for i in range(num_agents):
            self.actor_targets[i].load_state_dict(self.actors[i].state_dict())
            self.critic_targets[i].load_state_dict(self.critics[i].state_dict())

        self.replay_buffer = ReplayBuffer(capacity=MEMORY_SIZE, state_dim=state_dim, action_dim=action_dim, num_agents=num_agents)
        self.noises = [OUNoise(action_dim) for _ in range(num_agents)]
        for noise in self.noises:
            noise.reset()

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
    def get_action(self, states, noise_scale=0.1):
        actions = []
        for i in range(self.num_agents):
            state = states[i]
            if not isinstance(state, (list, tuple, np.ndarray)):  # Ensure state is a sequence
                state = [state]
            state = np.array(state, dtype=np.float32).reshape(1, -1)
            state = torch.FloatTensor(state).to(device)
            self.actors[i].eval()
            with torch.no_grad():
                action = self.actors[i](state).cpu().data.numpy()
            self.actors[i].train()
            noise = self.noises[i].noise() * noise_scale*(1-self.update_counter/10000)
            action = np.clip(action + noise, 0, 1)
            actions.append(action)
        return np.array(actions)

    def update(self):
        if self.replay_buffer.size < BATCH_SIZE:
            return
        self.update_counter += 1
        if self.update_counter %2 ==0:
            return
        batch = self.replay_buffer.sample(BATCH_SIZE)

        states = torch.FloatTensor(batch["states"]).to(device)
        actions = torch.FloatTensor(batch["actions"]).to(device)
        rewards = torch.FloatTensor(batch["rewards"]).to(device)
        next_states = torch.FloatTensor(batch["next_states"]).to(device)
        dones = torch.FloatTensor(batch["dones"]).to(device)

        all_actions = actions.view(BATCH_SIZE, -1)
        all_next_actions = []
        for i in range(self.num_agents):
            next_action = self.actor_targets[i](next_states[:, i, :])
            all_next_actions.append(next_action)
        all_next_actions = torch.cat(all_next_actions, dim=1)

        for i in range(self.num_agents):
            current_Q = self.critics[i](states.view(BATCH_SIZE, -1), all_actions)
            next_Q = self.critic_targets[i](next_states.view(BATCH_SIZE, -1), all_next_actions.detach())
            target_Q = rewards[:, i] + (1 - dones[:, i]) * GAMMA * next_Q.view(-1)
            critic_loss = nn.MSELoss()(current_Q.view(-1), target_Q)

            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            self.critic_optimizers[i].step()

            # Update Actor
            predicted_actions = []
            for j in range(self.num_agents):
                predicted_action = self.actors[j](states[:, j, :])
                predicted_actions.append(predicted_action)
            predicted_actions = torch.cat(predicted_actions, dim=1)
            # print(f"predicted_actions shape: {predicted_actions.shape}")
            actor_loss = -self.critics[i](states.view(BATCH_SIZE, -1), predicted_actions).mean()

            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()
            # Update target networks
        for i in range(self.num_agents):
            self.soft_update(self.actor_targets[i], self.actors[i], TAU)
            self.soft_update(self.critic_targets[i], self.critics[i], TAU)









