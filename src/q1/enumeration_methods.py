'''
enumeration_methods.py

This file contains methods for implementing enumeration and value/policy iteration for the shuttle problem.
'''

import yaml
import numpy as np
import seaborn as sns
from pathlib import Path

# Defining the Path to the config
CONFIG = Path("/nfs/home/yqy4012/AdvancedAlgos/HW1/config.yaml")

# Opening the Config
with open(CONFIG, "r") as f:
    config = yaml.safe_load(f)
q1 = config["question_1"]

### Helpers
def helper(K: int = q1["K"], cf: float = q1["c_f"], ch: float = q1["c_h"], T: float = q1["time_period"], discount_rate: float = q1["discount_rate"]):
    '''
    Selection of action 1 means sending the shuttle while the selection of action 0 means no shuttle is sent. 
    '''
    ### Consts
    arrivals = [1, 2, 3, 4, 5]
    station_capacity = q1["station_capacity"]
    
    ### Transition Matrices - 201x201
    prob_0, prob_1 = np.zeros((station_capacity + 1, station_capacity + 1)), np.zeros((station_capacity + 1, station_capacity + 1)) # plus 1 since we also account for the state of 0.

    for state in range(station_capacity + 1):
        for action in (0, 1):
            s_next = max(state - K, 0) if action == 1 else state # the next state should have at least 0 people after a dispatch
            for a in arrivals:
                s_after = min(s_next + a, station_capacity) # Following the dispatch, we may get new arrivals and those new arrivals cannot overflow the station capacity.
                if action == 0:
                    prob_0[state, s_after] += 0.2 # This summing of the current value captures potential overflow. For instance say we are at 199. Probability of s_after being 200 is 100%. This adds 0.2 5 times, getting to a probability of 1. 
                if action == 1:
                    prob_1[state, s_after] += 0.2 # We only add fixed 0.2 since the arrivals are uniformly distributed. 

    ### Reward - 2 x 201
    r = np.zeros((2, station_capacity + 1)) # A row for each action and the columns are the number of different states
    for state in range(station_capacity + 1):
        r[0, state] = ch * state
        r[1, state] = cf + ch * max(state - K, 0)

    return prob_0, prob_1, r
###

def enumeration(K: int = q1["K"], cf: float = q1["c_f"], ch: float = q1["c_h"], T: float = q1["time_period"], discount_rate: float = q1["discount_rate"]):
    '''
    Implementation of backwards enumeration. This method does not rely on a policy pi. We assume that V_T is the terminating function for the final state. Thus, we step through enumeration beginning 
    at V_(T-1). Thus, V_T = 0. Furthermore, the capacity at the station is capped at 200.

    The state is the number of people waiting at the station while the action space is either 0 (no dispatch) or 1 (dispatch). As such, we must construct two separate transition matrices, one for 
    each possible action. 

    The Bellman Equation: V_t(s_t) = max_(a_t) [ reward(s_t, a_t) + discount_factor * E[V_(t+1)(s_(t+1) | s_t)] ]

    Returns:
        V - The optimal Value Function at T = 0
        policy - The optimal policy at each state
    '''

    ### Initializing the probability and reward matrices
    prob_0, prob_1, r = helper(K, cf, ch)
    station_capacity = q1["station_capacity"]
    
    ### Bellman Equation
    # Value Function Results
    V = np.zeros(station_capacity + 1) # To be treated as column vector: 201x1
    policy = np.zeros((T, station_capacity + 1)) # Rows are timestep and columns are the different states. So for a given time, we can find the optimal policy for that state.
    
    for time in reversed(range(T)):        
        value_0 = r[0] + discount_rate * prob_0 @ V
        value_1 = r[1] + discount_rate * prob_1 @ V
        
        V = np.minimum(value_0, value_1) # minimum since the "reward" is actually a cost and we want to minimize that. 
        policy[time] = (value_1 < value_0) # When is dispatching cheaper, since dispatching is 1

    # Returns
    return V, policy

def value_iteration(K: int = q1["K"], cf: float = q1["c_f"], ch: float = q1["c_h"], T: float = q1["time_period"], discount_rate: float = q1["discount_rate"], tolerance=1e-8):
    '''
    Implementation of the value iteration method to solve the shuttle problem. 
    '''
    
    ### Initializing the probability and reward matrices
    prob_0, prob_1, r = helper(K, cf, ch)
    station_capacity = q1["station_capacity"]
    V = np.zeros(station_capacity + 1) # To be treated as column vector: 201x1; this is the starting point
    
    ### Algorithm
    while True:
        value_0 = r[0] + discount_rate * prob_0 @ V
        value_1 = r[1] + discount_rate * prob_1 @ V
        V_new = np.minimum(value_0, value_1)

        delta = np.max(np.abs(V_new - V))
        V = V_new
        if delta < tolerance:
            break

    # Greedy policy from converged V
    value_0 = r[0] + discount_rate * prob_0 @ V # We compute this because we break the loop before we can recalculate these values with the converged 
    value_1 = r[1] + discount_rate * prob_1 @ V
    policy = (value_1 < value_0)

    return V, policy

def policy_iteration(K: int = q1["K"], cf: float = q1["c_f"], ch: float = q1["c_h"], T: float = q1["time_period"], discount_rate: float = q1["discount_rate"], tolerance=1e-8):
    '''
    Implementation of the policy iteration algorithm to solve the shuttle problem. 
    '''

    ### Initializing the problem
    prob_0, prob_1, r = helper(K, cf, ch)
    station_capacity = q1["station_capacity"]
    V = np.zeros(station_capacity + 1)
    policy = np.zeros(station_capacity + 1)

    ### Policy Evaluation
    while True:
        # Policy Evaluation
        transition_matrix = np.where(policy[:, None] == 1, prob_1, prob_0) # Reshape policy array into a column vector. We select an entire row either from prob_1 or prob_0
        rewards = np.where(policy == 1, r[1], r[0])
        while True:
            V_new = rewards + discount_rate * transition_matrix @ V
            delta = np.max(np.abs(V_new - V))
            V = V_new
            if delta < tolerance:
                break

        # Policy Improvement
        value_0 = r[0] + discount_rate * prob_0 @ V
        value_1 = r[1] + discount_rate * prob_1 @ V
        new_policy = (value_1 < value_0) # When dispatching is cheaper

        # Our stopping condition is when the policy does not change
        if np.array_equal(policy, new_policy):
            break
        policy = new_policy

    return V, policy







    




    