#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import random
import Environment_new
import Agent_new

'''
learning_rate = 0.9
learning_step = 1000
discount_rate = 0.9
curiosity = 0.4
'''

trap_prob = 1.0 / 3
dimensions = (4,4)

random.seed(13)

env = Environment_new.Toy(dimensions,trap_prob)
agent = Agent_new.AgentQLearn(env)

agent.learn(test=False)
agent.test(10,debug=True)
# agent.print_policy((3,0))
