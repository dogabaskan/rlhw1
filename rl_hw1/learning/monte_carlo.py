""" Tabular MC algorithms
    Author: Doga Baskan
"""
from collections import defaultdict
import random
import math
from collections import deque
import numpy as np
import time



class TabularAgent:
    """ Based Tabular Agent class that inludes policies and evaluation function
    """

    def __init__(self, nact):
        self.qvalues = defaultdict(lambda: [0.0]*nact)
        self.nact = nact

    def greedy_policy(self, state, *args, **kwargs):
        """ Policy that returns the best action according to q values.
        """

        q_values = self.qvalues[state]
        best_action = np.argmax(q_values)
        
        return int(best_action)  # Ensure action is returned as an integer

    def e_greedy_policy(self, state, epsilon, *args, **kwargs):
        """ Policy that returns the best action according to q values with
        (epsilon/#action) + (1 - epsilon) probability and any other action with
        probability episolon/#action.
        """
        if random.random() < epsilon :
            return random.randint(0, self.nact -1 )
        
        return self.greedy_policy(state)
            

    def evaluate(self, env, render=False):
        """ Runs a single episode using the greedy policy to evaluate performance.
        """
        state = env.reset()  # Start new episode
        total_reward = 0
        done = False
        deliveries = 0  # Track how many deliveries are made

        while not done:
            if render:
                env.render()
            action = self.greedy_policy(state)  # Use greedy policy
            next_state, reward, done, _ = env.step(action)  # Take action
            total_reward += reward
            state = next_state  # Move to next state

            # Check for successful deliveries
            if reward > 0:  # If a delivery was made (reward > 0)
                deliveries += 1

            # If both deliveries are made, terminate the episode
            if deliveries == 2:
                total_reward = 2  # Set final reward to 2 for both deliveries
                done = True

        return total_reward


class MonteCarloAgent(TabularAgent):
    """ Tabular Monte Carlo Agent that updates q values based on MC method.
    """

    def __init__(self, nact):
        super().__init__(nact)

    def one_epsiode_train(self, env, policy, gamma, alpha):
        """ Single episode training function.
        Arguments:
            - env: Mazeworld environment
            - policy: Behaviour policy for the training loop
            - gamma: Discount factor
            - alpha: Exponential decay rate of updates

        Returns:
            episodic reward

        **Note** that in the book (Sutton & Barto), they directly assign the
        return to q value. You can either implmenet that algorithm (given in
        chapter 5) or use exponential decaying update (using alpha).
        """
        
        state = env.reset()
        done = False
        episode = []
        total_reward = 0
        deliveries = 0
        
        while not done:
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            total_reward += reward
            state = next_state
            
            if reward > 0:
                deliveries += 1
            
            if deliveries == 2:
                total_reward = 2
                done = True
        
        G = 0  
        visited = set()
        
        for state, action, reward in reversed(episode):
            G = gamma * G + reward  
            if (state, action) not in visited:
                visited.add((state,action))
                
                old_q = self.qvalues[state][action]
                self.qvalues[state][action] += alpha*(G-old_q)

        return total_reward
