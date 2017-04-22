#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import random
import Environment
import Agent

'''
learning_rate = 0.9
learning_step = 1000
discount_rate = 0.9
curiosity = 0.4
'''

trap_prob = 1.0 / 3
dimensions = (4,4)

random.seed(13)

env = Environment.Toy(dimensions,trap_prob)
env.debug = 1
agent = Agent.AgentQLearn(env)

agent.learn()
# agent.print_policy((3,0))
