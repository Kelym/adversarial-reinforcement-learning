#!/usr/bin/env python

'''
Skeleton from https://github.com/joacar/reinforcement-learning/blob/master/rl.py
'''

from __future__ import print_function
import numpy as np
import random
import environment
import agent

x_bound = 4
y_bound = 4
nSquares = x_bound*y_bound

# learning_rate = 0.9
learning_rate = 0.1
learning_step = 50 * nSquares * 4 * (2**4)
discount_rate = 0.9
curiosity = 0.4

#random.seed(13)

agent = agent.AgentQLearn((x_bound,y_bound),curiosity,discount_rate)

agent.learn(learning_rate, learning_step)
agent.print_policy()
