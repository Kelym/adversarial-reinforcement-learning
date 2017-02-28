from __future__ import print_function
import numpy as np
import random
import environment

class AgentQLearn(object):
	def __init__(self, dims, curiosity, discount_rate = 0.9):
		self.dims = dims
		self.env = environment.RandomMaze(dims)
		self.Q = [ [random.random()*0.5 for a in range(self.env.actions)] for s in range(self.env.states)]
		self.explore_chance = curiosity
		self.discount_rate = discount_rate

	def learn(self, learning_rate, steps):
		state = self.env.observe()
		change_maze = False
		for step in range(steps):
			if change_maze:
				self.env = environment.RandomMaze(self.dims)
				state = self.env.observe()
				change_maze = False
			if self.explore():
				 action = random.randrange(self.env.actions)
			else:
				 actions = self.Q[state]
				 action = actions.index(max(actions))

			reward = self.env.act(action)
			new_state = self.env.observe()

			if(self.env.debug > 2):
				 self.env.print_state(state)
				 self.env.print_action(action)
				 print(reward, ' ', new_state)

			next_action = self.Q[new_state].index(max(self.Q[new_state]))
			self.Q[state][action] += learning_rate*(reward + self.discount_rate*self.Q[new_state][next_action] - self.Q[state][action])
			state = new_state

			# we have reached the goal
			# start over with new random maze
			if reward == 200:
				change_maze = True

	def explore(self):
		return random.random() < self.explore_chance

	def print_policy(self):
		print(["{0:10}".format(i) for i in self.env.actions_name])
		for s in range(self.env.states):
			state = self.env.reverse_observe(s)
			self.env.print_state(state)
			action = self.Q[s].index(max(self.Q[s]))
			self.env.print_action(action)

			print('')
			print(["{0:10.2f}".format(i) for i in self.Q[s]])
