'''
modified from http://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
and https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/12%20-%20Deep%20Q%20Network/dqn13.py
'''

from __future__ import print_function
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from copy import deepcopy
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import torchvision.transforms as T

import Environment

Transition = namedtuple('Transition',
						('state', 'action', 'next_state', 'reward'))

# stores previously explored
class ReplayMemory(object):

	def __init__(self, capacity):
		self.capacity = capacity
		self.memory = []
		self.position = 0

	def push(self, *args):
		"""Saves a transition."""
		if len(self.memory) < self.capacity:
			self.memory.append(None)
		self.memory[self.position] = Transition(*args)
		self.position = (self.position + 1) % self.capacity

	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)

	def __len__(self):
		return len(self.memory)

class DQN(nn.Module):
	def __init__(self):
		super(DQN, self).__init__()
		# first convolutional layer designed to capture information about neighbors
		self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
		# good practice to use batch normalization before applying non-linear ReLu
		self.bn1 = nn.BatchNorm2d(16)
		self.fc1 = nn.Linear(16*3*3, 42)

		# four output neurons correspond to the four possible actions
		self.fc2 = nn.Linear(42,4)
		self.softmax = nn.softmax()

	def forward(self, x):
		x = F.relu(self.bn1(self.conv1(x)))
		# view flattens x so that it can be fed into FC layer
		x = F.relu(self.fc1(x.view(x.size(0), -1)))
		x = F.relu(self.fc2(x))
		return self.softmax(x)

class AgentQLearn():
	def __init__(self, env, curiosity=0.1, learning_rate=0.001, discount_rate = 0.9):
		self.env = env
		self.explore_chance = curiosity
		self.discount_rate = discount_rate

		# Initializes DQN and replay memory
		self.mem = ReplayMemory(capacity=10000)
		self.net = DQN()

        # using Huber Loss so that we're not as sensitive to outliers
		self.criterion = nn.SmoothL1Loss()
		self.optimizer = torch.optim.SGD(self.net.parameters(), lr=learning_rate)

	def learn(self, nEpochs=10, mini_batch_size=100, nSteps=10000):
		for epoch in range(nEpochs):
            self.env = self.env.RandomMaze()
			state = self.env.observe()

            restart = False
			for step in range(nSteps):
                # if we've reached the goal state in current maze,
                # restart with new random maze
                if restart:
    				self.env = self.env.new_maze()
    				state = self.env.observe()
    				restart = False

				# get next action; either from Q (network output) or exploration
				action = self.policy(state)
				reward, isGoalState = self.env.act(action)
				next_state = self.env.observe()

                self.mem.push(Transition(state,action,next_state,reward))

                if len(self.mem) >= mini_batch_size:
                    # Sample mini-batch transitions from memory
                    batch = self.mem.sample(mini_batch_size)
                    state_batch = np.vstack([trans.state for trans in batch])
                    # TODO may have to change action format to be ndarray
                    action_batch = np.vstack([trans.action for trans in batch])
                    reward_batch = np.vstack([trans.reward for trans in batch])
                    next_state_batch = np.vstack([trans.next_state for trans in batch])

                    # Forward + Backward + Optimize
                    self.net.zero_grad()
                    q_values = self.net(to_tensor(state_batch))
                    # Make volatile so that computational graph isn't affected
                    # by this batch of inputs
                    next_q_values = self.net(to_tensor(next_state_batch, volatile=True))
                    # change volatile flag back to false so that weight gradients will be calculated
                    next_q_values.volatile = False

                    target = to_tensor(reward_batch) + discount_factor * (next_q_values).max(1)[0]
                    loss = criterion(q_values.gather(1,to_tensor(action_batch).long().view(-1, 1)),
                                    target)
                    loss.backward()
                    optimizer.step()

                if isGoalState:
                    restart = True

            state = next_state

	# state is state as given by env; not necessarily in suitable format for network input
	def policy(self,state):
		if self.explore():
			action = random.randrange(self.env.actions)
		else:
			# Q-values from running network on current state
			q_values = self.net(state_to_input(state))
			# chooses best action
			# TODO confirm indexing
			action = q_values.data.max(1)[0][0,0]

		return action

	def explore(self):
		return random.random() < self.explore_chance

    # TODO can't enumerate over states
    '''
    def print_policy(self, start):
		self.env.set_position(start)
		s = self.env.observe()
		reward = 0
		print(["{0:10}".format(i) for i in self.env.actions_name])
		for s in range(self.env.states):
			self.env.print_state(s)
			action = self.Q[s].index(max(self.Q[s]))
			self.env.print_action(action)
			print('')
			print(["{0:10.2f}".format(i) for i in self.Q[s]])
    '''

# transforms numpy ndarray to Variable
def to_tensor(ndarray, volatile=False):
    return Variable(torch.from_numpy(ndarray), volatile=volatile).int()

# TODO function that transforms given state into format that aligns with
# input to network
def state_to_input(state):
	pass
