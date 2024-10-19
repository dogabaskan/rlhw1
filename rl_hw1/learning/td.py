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
        epsilon = args.init_eps
        episodic_rewards = []

        for episode in range(args.episodes):
            state = env.reset()
            done = False
            total_reward = 0


            while not done:
                action = policy(state, epsilon = epsilon)
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                
                if not done:
                    next_action = int(policy(next_state, epsilon)) 

                # Update Q-values using the agent's update method
                transition = (state, action, reward, next_state, next_action)
                self.update(transition, args.alpha, args.gamma)

                # Move to the next state and action
                state = next_state

            # Evaluate performance at the specified period
            if episode % args.evaluate_period == 0:
                eval_reward = self.evaluate(env)
                episodic_rewards.append(eval_reward)

            epsilon = max(args.final_eps, epsilon * args.eps_decay_rate)

        return episodic_rewards


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
        
    

        # Get the maximum Q-value for the next state (Q-learning is off-policy)
        next_q_value = max(self.qvalues[next_state])
        
        # Update Q-value for the current state-action pair
        current_q_value = self.qvalues[state][action]
        td_target = reward + gamma * next_q_value
        td_error = td_target - current_q_value

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

        # SARSA uses the Q-value of the next state-action pair (on-policy)
        next_q_value = self.qvalues[next_state][next_action]

        # Update Q-value for the current state-action pair
        current_q_value = self.qvalues[state][action]
        td_target = reward + gamma * next_q_value
        td_error = td_target - current_q_value

        self.qvalues[state][action] += alpha * td_error

        return td_error
