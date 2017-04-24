'''
modified from http://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
and https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/12%20-%20Deep%20Q%20Network/dqn13.py
'''

from __future__ import print_function
import math
import random
import numpy as np
import sys
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from copy import deepcopy
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable as var
import torch.nn.functional as F
import torchvision.transforms as T

if torch.cuda.is_available():
    import torch.backends.cudnn as cudnn
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    cudnn.benchmark = True

import Environment

Transition = namedtuple('Transition',
						('state', 'action', 'next_state', 'isNotTerminal', 'reward'))

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
	def __init__(self, dims):
		super(DQN, self).__init__()
		# first convolutional layer designed to capture information about neighbors
		self.conv1 = nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1)
		# good practice to use batch normalization before applying non-linear ReLu
		self.bn1 = nn.BatchNorm2d(4)
		x,y = dims
		#self.fc1 = nn.Linear(16*x*y, 16)
		self.fc1 = nn.Linear(4*x*y, 4)

		# four output neurons correspond to the four possible actions
		self.fc2 = nn.Linear(16,4)
		self.softmax = nn.Softmax()

	def forward(self, x):
		#print(x.data.size())
		x = F.relu(self.bn1(self.conv1(x)))
		#print(x.data.size())
		# view flattens x so that it can be fed into FC layer
		x = F.relu(self.fc1(x.view(x.size(0), -1)))
		#print(x.data.size())
		#x = F.relu(self.fc2(x))
		#return x
		return self.softmax(x)

class AgentQLearn():
	def __init__(self, env, curiosity=1, learning_rate=0.01, discount_rate = 0.9):
		self.env = env
		self.curiosity = curiosity
		self.discount_rate = discount_rate

		# Initializes DQN and replay memory
		self.mem = ReplayMemory(capacity=10000)
		self.net = DQN(self.env.dims())
		if torch.cuda.is_available():
			self.net = self.net.cuda()

		# using Huber Loss so that we're not as sensitive to outliers
		self.criterion = nn.SmoothL1Loss()
		if torch.cuda.is_available():
			self.criterion = self.criterion.cuda()
		#self.optimizer = torch.optim.SGD(self.net.parameters(), lr=learning_rate)
		self.optimizer = torch.optim.SGD(self.net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1E-4)

	def learn(self, nEpochs=3000, mini_batch_size=50, nSteps=1000, test=True, nTestMazes=100, end_curiosity=0.1):
		avg_rewards = []
		success_rates = []

		start = time.time()
		for epoch in range(nEpochs):
			self.env = self.env.new_maze()
			state = self.env.observe()

			if (epoch + 1) % 50 == 0:
				self.curiosity = max(self.curiosity-0.1, end_curiosity)
				

			'''
			initial_state = state.copy()
			for flag in initial_state:
				print(flag)
			initial_q_values = self.net(to_var(initial_state, volatile=True))
			print(initial_q_values)
			'''
			
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
				reward, square = self.env.act(action)
				next_state = self.env.observe()

				# state is terminal if trap or goal
				# note: for learning purposes, learning still happens on same maze
				# after trap is encountered
				isNotTerminal = not (self.env.isTrap(square) or self.env.isGoal(square))
				self.mem.push(state,action,next_state,isNotTerminal,reward)

				'''
				if self.env.isTrap(square) or self.env.isGoal(square):
					self.mem.push(state,action,None,reward)
				else:
					self.mem.push(state,action,next_state,reward)
				'''

				if len(self.mem) >= mini_batch_size:
					# Sample mini-batch transitions from memory
					batch = self.mem.sample(mini_batch_size)
					state_batch = np.array([trans.state for trans in batch])
					# TODO may have to change action format to be ndarray
					action_batch = np.array([trans.action for trans in batch])
					reward_batch = np.array([trans.reward for trans in batch])
					next_state_batch = np.array([trans.next_state for trans in batch])
					isNotTerminal_batch = np.array([int(trans.isNotTerminal) for trans in batch])

					# is 0 if state is terminal (trap or goal)
					non_terminal_mask = to_var(isNotTerminal_batch)

					'''
					print(next_state_batch)
					print(non_final_mask)
					sys.exit(0)
					'''

					'''
					# collects all next states that aren't terminal to be fed into model
					# volatile so that grad isn't calculated w.r.t. to this feed-forward
					non_terminal_next_states = to_var(np.array([s for s in next_state_batch
												if s is not None]),
												volatile=True)
					next_q_values = to_var(np.zeros((mini_batch_size,)))
					next_q_values[non_final_mask] = self.net(non_final_next_states).max(1)[0]
					notGoalFunc = lambda s: int(not self.env.isGoalState(s))
					next_NOTGoal_batch = np.array([notGoalFunc(s) for s in next_state_batch])
					'''

					# Forward + Backward + Optimize
					self.net.zero_grad()
					cur_input = to_var(state_batch, volatile=False)

					if torch.cuda.is_available():
						cur_input = cur_input.cuda()

					# feeds-forward state batch and collects the action outputs corresponding to the action batch
					q_values = self.net(cur_input).gather(1,to_var(action_batch).long().view(-1, 1))

					# Make volatile so that computational graph isn't affected
					# by this batch of inputs
					next_input = to_var(next_state_batch, volatile=True)
					if torch.cuda.is_available():
						next_input = next_input.cuda()
					
					next_max_q_values = self.net(next_input).max(1)[0].float()

					'''
					print(next_q_values.data)
					print(next_q_values.data.max(1))
					print(next_q_values.data.max(1)[0])
					'''
				
					# change volatile flag back to false so that weight gradients will be calculated
					next_max_q_values.volatile = False

					# only includes future q values if neither in goal state nor trap state
					target = to_var(reward_batch) + self.discount_rate * next_max_q_values * non_terminal_mask
					loss = self.criterion(q_values, target)

					self.optimizer.zero_grad()
					loss.backward()
					self.optimizer.step()

				if self.env.isGoal(square):
					assert self.env.isGoalState(next_state)
					restart = True

				state = next_state

			print('\nFinished epoch %d....' % epoch)

			if test:
				nSuccesses, avg_reward = self.test(nTestMazes)
				print('Success rate: %d / %d\nAvg. Reward: %f' 
					% (nSuccesses, nTestMazes, avg_reward))

				avg_rewards.append(avg_reward)
				success_rates.append(float(nSuccesses) / nTestMazes)

		end = time.time()

		print('\nLearning Time: %f' % (end - start))

		return avg_rewards, success_rates

		'''
		final_q_values = self.net(to_var(initial_state, volatile=True))
		print(final_q_values)
		'''
	def save_model(self,fname):
		torch.save(self.net.cpu().state_dict(),fname)

	# state is state as given by env; not necessarily in suitable format for network input
	def policy(self,state,can_explore=True):
		if can_explore and self.explore():
			action = random.randrange(self.env.actions)
		else:
			# Q-values from running network on current state
			# Brackets around state give it batch dimension (of size 1)
			# q_values = self.net(batch_to_input([state]))
			v = to_var(state, volatile=True)
			if torch.cuda.is_available():
				v = v.cuda()

			q_values = self.net(v)
			# chooses best action
			action = q_values.data.max(1)[1][0,0]

		return int(action)

	def explore(self):
		return random.random() < self.curiosity

	def test(self, nMazes, debug=False):
		avg_reward = 0.0
		nSuccesses = 0
		max_nSteps = self.env.xbound * self.env.ybound
		for test in range(nMazes):
			env = self.env.new_maze()
			state = env.observe()
			cum_reward = 0.0

			if debug:
				print('Initial State:')
				env.print_state()
			
			for step in range(max_nSteps):

				action = self.policy(state,can_explore=False)
				reward, square = env.act(action)
				cum_reward += reward

				if debug:
					print('\n-------------------------------\n')
					print('Step %d:\nPrev Action: ' % step, end='')
					env.print_action(action)
					print()
					env.print_state()
					print('Cumulative reward: %f' % cum_reward)

					input('Press enter....')

				if env.isGoal(square):
					nSuccesses += 1
					break

				if env.isTrap(square):
					break

				state = env.observe()

			avg_reward += cum_reward

		return nSuccesses, avg_reward / nMazes


			

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
# transforms numpy ndarray to Tensor
def to_tensor(ndarray):
	
	return torch.from_numpy(ndarray).float()

# transforms numpy ndarray to Variable for input into network
def to_var(ndarray, volatile=False):
	tensor = to_tensor(ndarray)
	if tensor.dim() == 3:
		tensor.unsqueeze_(0)

	v = var(tensor, volatile=volatile)
	if torch.cuda.is_available():
		v = v.cuda()

	return v

# TODO function that transforms given batch into format that aligns with
# input to network
def batch_to_input(batch):

	return to_tensor(np.array(batch), volatile=True)

