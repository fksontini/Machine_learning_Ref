#!/usr/bin/env python3
""" Q Learning """
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """ Use epsilon-greedy to determine the next action """
    action = 0
    p = np.random.uniform(0, 1)
    if p < epsilon:
        action = np.random.randint(4)
    else:
        action = np.argmax(Q[state, :])
    return action


def train(
        env,
        Q,
        episodes=5000,
        max_steps=100,
        alpha=0.1,
        gamma=0.99,
        epsilon=1,
        min_epsilon=0.1,
        epsilon_decay=0.05):
    """ Perform Q-learning """
    total_rewards = []
    for episode in range(episodes):
        state = env.reset()
        total = 0
        t = 0
        while t < max_steps:
            action = epsilon_greedy(Q, state, epsilon)
            state2, reward, done, info = env.step(action)
            if done is True and reward == 0:
                reward = -1
            predict = Q[state, action]
            target = reward + gamma * np.max(Q[state2, :])
            Q[state, action] += alpha * (target - predict)
            total += reward
            state = state2
            if done:
                break
            t += 1
        epsilon = (epsilon - min_epsilon) * \
            np.exp(-epsilon_decay * episode) + min_epsilon
        total_rewards.append(total)
    return Q, total_rewards
