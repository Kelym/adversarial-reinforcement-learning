#!/usr/bin/env python

'''
Skeleton from https://github.com/joacar/reinforcement-learning/blob/master/rl.py
'''

from __future__ import print_function
import numpy as np
import random
import Environment
import Agent

learning_rate = 0.9
learning_step = 1000
discount_rate = 0.9
curiosity = 0.4

random.seed(13)

maze = 'Mazes/2'
start = (3,0)

env = Environment.Toy(maze, start)
env.debug = 1
agent = Agent.AgentQLearn(env, curiosity,discount_rate)

agent.learn(learning_rate, learning_step)
agent.print_policy((3,0))
