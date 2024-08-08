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
    probs = [0.7,0.1,0.1,0.1]
    transition_map = {}
    
    for i in range(rows):
        for j in range(cols):
            if initial_board[i][j] != '#':
                state = (i, j)
                transition_map[state] = {}
                for action in range(4):
                    transitions = []
                    for k in range(4):
                        
                        if k == 0:
                            next_state = (i-1, j)  
                        elif k == 1:
                            next_state = (i+1, j)  
                        elif k == 2:
                            next_state = (i, j-1)  
                        elif k == 3:
                            next_state = (i, j+1)  
                        
                        if (0 <= next_state[0] < rows and 0 <= next_state[1] < cols and 
                            initial_board[next_state[0]][next_state[1]] != '#'):
                            reward = 1 if initial_board[next_state[0]][next_state[1]] == "@" else 0
                            terminal = initial_board[next_state[0]][next_state[1]] == "@"
                            transitions.append((probs[k], next_state, reward, terminal))
                        else:
                            reward = 0
                            transitions.append((probs[k], state, reward, False))
                            
                    transition_map[state][action] = transitions
    return transition_map

    
    
    
    
    
    
    
