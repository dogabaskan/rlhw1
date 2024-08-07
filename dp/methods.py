""" This module includes DP algorithms.
    Author: Your-name / Your-number
"""


class DPAgent():
    r""" Base Dynamic Programming class. DP methods requires the transition
    map in order to optimize policies. This class provides the policy and one
    step policy evaluation as well as policy improvement. It serves as a
    base class for Policy Iteration and Value Iteration algorithms.
    """

    def __init__(self, nact, transitions_map, init_value=0.0):
        self.nact = nact
        self.transitions_map = transitions_map
        self.values = {s: init_value for s in self.transitions_map.keys()}
        self.policy_dist = {s: [1.0/nact] *
                            nact for s in self.transitions_map.keys()}

    def policy(self, state):
        """ Policy pi that returns the action with the highest Q value.
        You can use "policy_dist" that keeps probability values for each
        action at any state. You may have a stochastic policy(random.choice)
        or a determistic one(argmax). Your choice. But for this homework we
        will be using deterministic policy. Therefore, at each state only one
        action will be possible with probability 1.(Initially its stochastic)
        Example policy_dist:
            policy_dist[S] = [1, 0, 0, 0]

        Arguments:
            - state: State/observation of the environment

        Returns:
            action for the given state
        """
        return max(range(self.nact),key = lambda a : self.policy_dist[state][a])
        
    
    def one_step_policy_eval(self, gamma=0.95):
        """ One step policy evaluation.
        You can follow the pseudocode given in the "Reinforcement Learing
        Book (Sutton & Barta) chapter 4.1"
        Remember to return delta: Maximum change among all the states

        Arguments:
            - gamma: Discount factor

        Return:
            delta
        """
        delta = 0
        for state in self.values:
            v = self.values[state]
            new_val = 0
            for action, action_prob in enumerate(self.policy_dist[state]):
                for prob, next_state, reward, done in self.transitions_map[state][action]:
                    new_val += action_prob * prob * (reward + gamma * self.values[next_state])
            delta = max(delta, abs(v - new_val))
            self.values[state] = new_val
        return delta
    
    def policy_improvement(self, gamma=0.95):
        """Policy impovement updates the policy according to the most recent
        values. You can follow the ideas given in the "Reinforcement Learing
        (Sutton & Barta) chapter 4.2. Basically, look one step ahead to choose
        the best action.
        Remember to return a boolean value to state if the policy is stable
        Also note that, improved policy must be deterministic. So that, at
        each state only one action is probable(like onehot vector).
        Example policy:
            policy_dist[S] = [0, 1, 0, 0]

        Arguments:
            - gamma: Discount factor

        Return:
            a boolean value stating if stability is reached
        """
        policy_stable = True
        for state in self.transitions_map :
            old_action = self.policy(state)
            q_val = [sum(prob* (reward + gamma * self.values[next_state]) 
                for prob, next_state, reward, terminal in self.transitions_map[state][action]) for action in range(self.nact)]
            best_action = max(range(self.nact), key=lambda a:q_val[a])
            best_action_dist = [0.0] * self.nact
            best_action_dist[best_action] = 1.0
            self.policy_dist[state] = best_action_dist
            
            if old_action != best_action:
                policy_stable = False
                
        return policy_stable
            

class PolicyIteration(DPAgent):
    r""" Policy Iteration algorithm that first evaluates the
    values until they converge within epsilon range, then
    updates the policy and repeats the process until the
    policy no longer changes.
    """

    def __init__(self, transitions_map, nact=4):
        super().__init__(nact, transitions_map)

    def optimize(self, gamma, epsilon=0.05, n_iteration=1):
        """ This is the main function where you implement PI.
        Simply use "one_step_policy_eval" and "policy_improvement" methods.
        Itearte as "n_iteration" times in a loop.
        Arguments:
            - gamma: Discount factor
            - epsilon: convergence region of value evaluation
            - n_iteration: Number of iterations
        This should not take more than 10 lines.
        """
        for _ in range (n_iteration):
            while True: 
                delta = self.one_step_policy_eval(gamma)
                if delta < epsilon:
                    break
                if self.policy_improvement(gamma):
                    break


class ValueIteration(DPAgent):
    r""" Value Iteration algorithm iteratively evaluates
    the values and updates the policy until the values
    converges within epsilon range.
    """

    def __init__(self, transitions_map, nact=4):
        super().__init__(nact, transitions_map)

    def optimize(self, gamma, n_iteration=1):
        """ This is the main function where you implement VI.
        Simply use "one_step_policy_eval" and "policy_improvement" methods.
        Itearte as "n_iteration" times in a loop.
        Arguments:
            - gamma: Discount factor
            - n_iteration: Number of iterations
        This should not take more than 5 lines.
        """
        for _ in range(n_iteration):
            self.one_step_policy_eval(gamma)
            self.policy_improvement(gamma)
            
            