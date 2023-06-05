import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import matplotlib.pyplot as plt
from tqdm import tqdm
import LSPI
import basisFunctions

# hyper-parameters
BATCH_SIZE = 128
LR = 0.01
GAMMA = 0.90
EPISILO = 0.9
MEMORY_CAPACITY = 2000
Q_NETWORK_ITERATION = 100

env = gym.make("CartPole-v0")
env = env.unwrapped
NUM_ACTIONS = env.action_space.n
NUM_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample.shape

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(NUM_STATES, 50)
        self.fc1.weight.data.normal_(0,0.1)
        self.fc2 = nn.Linear(50,30)
        self.fc2.weight.data.normal_(0,0.1)
        self.out = nn.Linear(30,NUM_ACTIONS)
        self.out.weight.data.normal_(0,0.1)

    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        action_prob = self.out(x)
        return action_prob

class DQN():
    def __init__(self):
        super(DQN, self).__init__()
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, NUM_STATES * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        if np.random.randn() <= EPISILO:
            action_value = self.eval_net.forward(state)
            action = torch.max(action_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        else:
            action = np.random.randint(0,NUM_ACTIONS)
            action = action if ENV_A_SHAPE ==0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1


    def learn(self):

        if self.learn_step_counter % Q_NETWORK_ITERATION ==0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter+=1

        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        batch_memory = self.memory[sample_index, :]
        batch_state = torch.FloatTensor(batch_memory[:, :NUM_STATES])
        batch_action = torch.LongTensor(batch_memory[:, NUM_STATES:NUM_STATES+1].astype(int))
        batch_reward = torch.FloatTensor(batch_memory[:, NUM_STATES+1:NUM_STATES+2])
        batch_next_state = torch.FloatTensor(batch_memory[:,-NUM_STATES:])

        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        q_next = self.target_net(batch_next_state).detach()
        q_target = batch_reward + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def reward_func(env, x, x_dot, theta, theta_dot):
    r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.5
    r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
    reward = r1 + r2
    return reward

def main(episodes):
    dqn = DQN()
    episodes = episodes
    num_steps = [0]
    episodes_list = [0]
    for i in range(episodes):
        state = env.reset()
        ep_reward = 0
        steps = 0
        episodes_list.append(i + 1)
        while True:
            steps += 1
            env.render()
            action = dqn.choose_action(state)
            next_state, _ , done, info = env.step(action)
            x, x_dot, theta, theta_dot = next_state
            reward = reward_func(env, x, x_dot, theta, theta_dot)

            dqn.store_transition(state, action, reward, next_state)
            ep_reward += reward

            if dqn.memory_counter >= MEMORY_CAPACITY:
                dqn.learn()
                if done:
                    print("episode: {} , the episode reward is {}".format(i, round(ep_reward, 3)))
            if done:
                break
            state = next_state
        num_steps.append(steps)
    plt.plot(episodes_list[::50], num_steps[::50], color='blue', linewidth=2.0, linestyle='--', label="DQN")
    plt.ylim((0, 3000))
    plt.xlim((0, episodes + 1))

def lspi_func(episodes):
    env = gym.make("CartPole-v0")

    bfs = basisFunctions.get_cartpole_basis_functions_quadratic_v2()

    gamma = 0.95
    epsilon = 0.01
    k = len(bfs)

    w = np.zeros(k)
    w_est, w0 = LSPI.LSPI(bfs, gamma, epsilon, w, env, method="discrete", n_trial_samples=1000, n_timestep_samples=6)

    env._max_episode_steps = 3000
    num_episodes = episodes

    method = "discrete"

    num_steps = [0]
    episodes_list = [0]
    for i_episode in range(num_episodes):
        episodes_list.append(i_episode + 1)
        observation = env.reset()
        print("--------")
        t = 0
        actions = []
        while True:
            t += 1
            env.render()
            action = LSPI.get_policy_action(env.env.state, w_est, bfs, env, method=method)

            # action = env.action_space.sample()
            if method == "continuous":
                action = [action[0]]
            observation, reward, done, info = env.step(action)
            # print observation
            if done:
                print("reward:", reward)
                num_steps.append(t)
                print("Episode finished after {} timesteps".format(t + 1))
                break
    plt.plot(episodes_list[::50], num_steps[::50], color='purple', linewidth=2.0, linestyle='--', label="LSPI")
    plt.ylim((0, 3000))
    plt.xlim((0, num_episodes + 1))
    plt.show()
        

if __name__ == '__main__':
    episodes = 300
    main(episodes)
    lspi_func(episodes)