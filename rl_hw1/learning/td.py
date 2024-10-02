""" Tabular TD methods
    Author: Your-name / Your-number
"""
from collections import defaultdict
from collections import namedtuple
import random
import math
import numpy as np
import time
from collections import deque

from .monte_carlo import TabularAgent


class TabularTDAgent(TabularAgent):
    """ Base class for Tabular TD agents for shared training loop.
    """

    def train(self, env, policy, args):
        """ Training loop for tabular td agents.
        Initiate an episodic reward list. At each episode decrease the epsilon
        value exponentially using args.eps_decay_rate within the boundries of
        args.init_eps and args.final_eps. For every "args._evaluate_period"'th
        step call evaluation function and store the returned episodic reward
        to the list.

        Arguments:
            - env: Warehouse environment
            - policy: Behaviour policy to be used in training(not in
            evaluation)
            - args: namedtuple of hyperparameters

        Return:
            - Episodic reward list of evaluations (not the training rewards)

        **Note**: This function will be used in both Sarsa and Q learning.
        **Note** that: You can also implement you own answer to question 10.
        """
        episodic_rewards = []
        epsilon = args.init_eps
        
        for episode in range(args.episodes):
            # Decay epsilon based on args.eps_decay_rate
            epsilon = max(args.final_eps, epsilon * args.eps_decay_rate)

            # Run one episode using the policy (e.g., epsilon-greedy)
            reward = self.one_episode_train(env, policy, epsilon, args.gamma, args.alpha)

            if (episode + 1) % args._evaluate_period == 0:
                eval_reward = self.evaluate(env)
                episodic_rewards.append(eval_reward)

        return episodic_rewards
    
    def one_episode_train(self, env, policy, epsilon, gamma, alpha):
        """ Conducts one episode of training using the specified policy.

        Arguments:
            - env: Environment in which to train the agent
            - policy: Policy to use for selecting actions
            - epsilon: Current epsilon value for epsilon-greedy policy
            - gamma: Discount factor
            - alpha: Learning rate

        Return:
            - total_reward: Total reward accumulated during the episode
        """
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = policy(state, epsilon)  # Use the provided policy
            next_state, reward, done, _ = env.step(action)  # Execute action
            
            # Update Q-values (for Q-learning or Sarsa)
            best_next_action = np.argmax(self.qvalues[next_state])
            td_target = reward + gamma * self.qvalues[next_state][best_next_action]
            td_delta = td_target - self.qvalues[state][action]
            self.qvalues[state][action] += alpha * td_delta  # Update the Q-value

            total_reward += reward
            state = next_state

        return total_reward


class QAgent(TabularTDAgent):
    """ Tabular Q leanring agent. Update rule is based on Q learning.
    """

    def __init__(self, nact):
        super().__init__(nact)

    def update(self, transition, alpha, gamma):
        """ Update values of a state-action pair based on the given transition
        and parameters.

        Arguments:
            - transition: 5 tuple of state, action, reward, next_state and
            next_action. "next_action" will not be used in q learning update.
            It is there to be compatible with SARSA update in "train" method.
            - alpha: Exponential decay rate of updates
            - gamma: Discount ratio

        Return:
            temporal diffrence error
        """
        state, action, reward, next_state, _ = transition
        max_next_q = max(self.qvalues[next_state])  # Off-policy: use max a' of Q(s', a')
        td_error = reward + gamma * max_next_q - self.qvalues[state][action]
        self.qvalues[state][action] += alpha * td_error
        return td_error

class SarsaAgent(TabularTDAgent):
    """ Tabular Sarsa agent. Update rule is based on
    SARSA(State Action Reward next_State, next_Action).
    """

    def __init__(self, nact):
        super().__init__(nact)

    def update(self, trans, alpha, gamma):
        """ Update values of a state-action pair based on the given transition
        and parameters.

        Arguments:
            - transition: 5 tuple of state, action, reward, next_state and
            next_action.
            - alpha: Exponential decay rate of updates
            - gamma: Discount ratio

        Return:
            temporal diffrence error
        """
        state, action, reward, next_state, next_action = trans
        td_error = reward + gamma * self.qvalues[next_state][next_action] - self.qvalues[state][action]
        self.qvalues[state][action] += alpha * td_error
        return td_error
