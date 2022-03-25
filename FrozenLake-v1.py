import numpy as np
import gym
env = gym.make('FrozenLake-v1')

def value_iteration(env):
    num_iterations = 5000
    thre = 1e-20
    gamma = 0.9
    value_table = np.zeros(env.observation_space.n)

    for i in range(num_iterations):
        updated_value_table = np.copy(value_table)
        for s in range(env.observation_space.n):
            q_values = [sum([prob*(r+gamma*updated_value_table[s_])for prob,s_,r,_ in env.P[s][a]])for a in range(env.action_space.n)]
            value_table[s] = max(q_values)
        if(np.sum(np.fabs(updated_value_table - value_table)) <= thre): # np.fabs: 절댓값
            break
    return value_table

def policy_iteration(value_table):
    gamma = 0.9
    policy = np.zeros(env.observation_space.n)

    for s in range(env.observation_space.n):
        q_values = [sum([prob*(r+gamma*value_table[s_])for prob,s_,r,_ in env.P[s][a]])for a in range(env.action_space.n)]
        policy[s] = np.argmax(np.array(q_values))
    return policy

optimal_value_function = value_iteration(env)
optimal_policy = policy_iteration(optimal_value_function)
print(optimal_value_function)
print(optimal_policy)

obs = env.reset()   # obs = observation
episode_reward = 0.0

for _ in range(100):
    action = np.int_(optimal_policy[obs])
    obs,reward,done,info = env.step(action)
    env.render()
    episode_reward += reward
    if done:
        print('reward:', episode_reward)
        episode_reward = 0.0
        obs = env.reset()
