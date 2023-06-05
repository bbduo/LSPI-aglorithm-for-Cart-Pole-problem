import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import LSPI
import basisFunctions

env = gym.make("CartPole-v0")

bfs = basisFunctions.get_cartpole_basis_functions_quadratic_v2()

gamma = 0.93
epsilon = 0.01
k = len(bfs)

w = np.zeros(k)
w_est, w0 = LSPI.LSPI(bfs, gamma, epsilon, w, env, method = "discrete", n_trial_samples=1000, n_timestep_samples=6)
print (w_est)

env._max_episode_steps = 100000

method = "discrete"
#method = "continuous"

num_steps = []
for i_episode in range(100):
    observation = env.reset()
    print ("--------")
    t = 0
    actions = []
    while True:
        t += 1
        env.render()
        action = LSPI.get_policy_action(env.env.state, w_est, bfs, env, method = method)
        if method == "continuous":
            action = [action[0]]
        observation, reward, done, info = env.step(action)
        if done:
            print ("reward:", reward)
            num_steps.append(t)
            print("Episode finished after {} timesteps".format(t+1))
            break
plt.hist(num_steps)
print(np.mean(num_steps))