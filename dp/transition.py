""" Produce the necessary transition map for dynamic programming.
    Author: Your-name / Your-number
"""
from collections import defaultdict
import numpy as np


def make_transition_map(initial_board):
    """ Return the transtion map for passable(other than wall cells) states.
    In a state S an action A is chosen, there are four possibilities:
    - Intended action can be picked
    - 3 of the remaning action can be picked by the environment.
    Structure of the map:

    map[S][A] -> [
        (p_0, n_s, r_0, t_0), # Quad tuple of transition for the action 0
        (p_1, n_s, r_1, t_1), # Quad tuple of transition for the action 1
        (p_2, n_s, r_2, t_2), # Quad tuple of transition for the action 2
        (p_3, n_s, r_3, t_3), # Quad tuple of transition for the action 3
    ]

    p_x denotes the probability of transition by action "x"
    r_x denotes the reward obtained during the transition by "x"
    t_x denotes the termination condition at the new state(next state)
    n_s denotes the next state

    S denotes the space of all the non-wall states
    A denotes the action space which is range(4)
    So each value in map[S][A] is a length 4 list of quad tuples.


    Arguments:
        - initial_board: Board of the Mazeworld at initialization

    Return:
        transition map
    """

    rows = len(initial_board)
    cols = len(initial_board[0])

    # Identify passable states, which are not walls ('#' or 35 in ASCII)
    passable_states = [(i, j) for i in range(rows) for j in range(cols) if initial_board[i][j] != 35]

    transition_map = {}

    # Directions for actions: 0: up, 1: down, 2: left, 3: right
    directions = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

    for state in passable_states:
        transition_map[state] = {}
        
        for action in range(4):
            transitions = []
            
            for k in range(4):  # Explore all possible outcomes (including stochastic actions)
                direction = directions[k]
                new_state = (state[0] + direction[0], state[1] + direction[1])

                # Check if the new state is within bounds and passable
                if new_state in passable_states:
                    reward = 1.0 if initial_board[new_state[0]][new_state[1]] == 64 else 0.0
                    terminal = initial_board[new_state[0]][new_state[1]] == 64
                else:
                    new_state = state  # Stay in the same state if the next one is not passable
                    reward = 0.0
                    terminal = False

                prob = 0.7 if k == action else 0.1
                transitions.append((prob, new_state, reward, terminal))

            transition_map[state][action] = transitions

    return transition_map