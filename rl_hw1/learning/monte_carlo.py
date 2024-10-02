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
            return int(random.randint(0, self.nact -1 ))
        
        return int(self.greedy_policy(state))
            

    def evaluate(self, env, render=False):
        """ Runs a single episode using the greedy policy to evaluate performance.
        """
        state = env.reset()  # Assuming the environment has reset()
        done = False
        total_reward = 0
        while not done:
            if render:
                env.render()
            action = self.greedy_policy(state)
            next_state, reward, done, _ = env.step(action)  # Execute action
            total_reward += reward
            state = next_state
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
        
        # Run the episode, collecting the state, action, and reward.
        while not done:
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state
        
        # Calculate returns and update Q-values
        G = 0  # Initialize the return
        for state, action, reward in reversed(episode):
            G = gamma * G + reward  # Discounted return
            if alpha is None:
                # First-visit MC method (direct return update)
                self.qvalues[state][action] = G
            else:
                # Incremental MC method with exponential decay
                self.qvalues[state][action] += alpha * (G - self.qvalues[state][action])
        
        # Return the episodic reward
        return sum([reward for _, _, reward in episode])
