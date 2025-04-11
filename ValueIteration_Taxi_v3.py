import numpy as np
import gymnasium as gym
import random
import math

# Create the Taxi environment (wrapped)
env = gym.make("Taxi-v3", render_mode="human")

# Unwrap to access transition probabilities for value iteration
env_unwrapped = env.unwrapped

# change-able parameters:
discount_factor = 0.8  # gamma
delta_threshold = 0.00001  # convergence threshold
epsilon = 1


def value_iteration(env, gamma, epsilon):
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    # Initialize the value function
    V = np.zeros(num_states)

    #Implement value iteration
    while True:
        delta = 0
        for state in range(num_states):
            A = np.zeros(num_actions)
            for action in range(num_actions):
                for prob, next_state, reward, terminated in env.P[state][action]:
                    A[action] += prob * (reward + gamma * V[next_state])
            max_value = max(A)
            delta = max(delta, abs(max_value - V[state]))
            V[state] = max_value
        if delta < epsilon:
            break

    # For each state, the policy will tell you the action to take
    policy = np.zeros(num_states, dtype=int)

    #extract the optimal policy from value function.
    for state in range(num_states):
        A = np.zeros(num_actions)
        for action in range(num_actions):
            for prob, next_state, reward, terminated in env.P[state][action]:
                A[action] += prob * (reward + gamma * V[next_state])
        policy[state] = np.argmax(A)

    return policy, V


# Run Value Iteration
policy, V = value_iteration(env_unwrapped, discount_factor, delta_threshold)

# resetting the environment and executing the policy
state, _ = env.reset()
done = False

max_steps = 100
for step in range(max_steps):

    # Getting max value against that state, so that we choose that action
    
    action = policy[state]
    new_state, reward, terminated, truncated, info = env.step(action)
    env.render()

    if terminated or truncated:
        print("Episode finished in", step, "steps.")
        break

    state = new_state

env.close()
