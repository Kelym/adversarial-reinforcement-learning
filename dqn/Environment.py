from __future__ import print_function
import numpy as np
import random

'''
The Environment Class shall all respond to two calls
  - act
	Do an action to the internal current state
	Calculate the next state
	And return the reward
  - observe
	Return the full state representation
Extra
  - read_maze
  - print_state
  - print_action
  - set_position

The following is a naive environment, an m x n maze generated randomly
'''

class Toy(object):
	# trap_prob is likelihood that given square is a trap
	def __init__(self, dims, trap_prob=-1):
		self.xbound, self.ybound = dims
		self.nSquares = self.xbound * self.ybound

        # total number of states; irrelevant now
		#self.states = self.nSquares * (2**4) * 4

		self.actions = 4
		self.movement = [[0,1],[1,0],[0,-1],[-1,0]]
		self.actions_name = ['Right','Down','Left','Up']

		if trap_prob == -1:
			# if not specified trap_prob is between 0 and 1/3
			self.trap_prob = random.uniform(0,0.33)
		else:
			self.trap_prob = trap_prob

		traps = np.random.binomial(1,self.trap_prob,self.nSquares)
		self.maze = np.array(['X' if traps[i] else '' for i in range(self.nSquares)])

		randoms = np.random.randint(self.nSquares, size=2)
		start = randoms[0]
		s_x = start / self.ybound
		s_y = start % self.ybound

		# initial starting position
		self.pos = (s_x, s_y)

		self.maze[start] = 'S'

		goal = randoms[1]
		self.maze[goal] = 'O'

		self.maze = self.maze.reshape(dims)

		self.debug = 0

    def new_maze(self):
        return Toy((self.xbound, self.ybound), self.trap_prob)

	def act(self, action):

		x = self.state[0]
		y = self.state[1]

		new_x = x + self.movement[action][0]
		new_y = y + self.movement[action][1]

		if(self.debug > 3):
			print(x,' ',y,' + ', self.actions_name[action], ' -> ',new_x, ' ', new_y)

		if(new_x < 0 or new_x >= self.xbound or new_y < 0 or new_y >= self.ybound):
			# Out of maze
			return -100, False
		if(self.maze[new_x][new_y] == 'X'):
			# Block
			return -100, False

		self.state = (new_x, new_y)
		if(self.maze[new_x][new_y] == 'O'):
			# Target
			return 100, True
		# Punish for time passed
		return -0.5, False

	# TODO change state output to be array representation (e.g. self.maze)
	def observe(self):
		# The method will produce a single number representing distinct states
		return self.state[0]*self.ybound + self.state[1];

	def print_state(self, state):
		print(state/self.ybound, state%self.ybound, end=' ')

	def print_action(self, action):
		print(self.actions_name[action], end=' ')

	def set_position(self,pos):
		self.pos = pos
