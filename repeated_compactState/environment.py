from __future__ import print_function
import numpy as np
import random

'''
The RandomMaze Class shall all respond to two calls
  - act
	Do an action to the internal current state
	Calculate the next state
	And return the reward
  - observe
	Return the full state representation
Extra
  - print_state
  - print_action
  - set_position

The following is a naive environment, an m x n maze generated randomly
'''

class RandomMaze(object):
	# trap_prob is likelihood that given square is a trap
	def __init__(self, dims, trap_prob=-1):
		self.xbound, self.ybound = dims
		self.nSquares = self.xbound * self.ybound
		# state is now tuple of (cur_xy, knowledge of each neighbor, "least trappiest quadrant")
		# "least trappiest quadrant" is quadrant (cur_xy as origin) with least traps
		self.states = self.nSquares * (2**4) * 4
		# we now have a state-independent KB:
		# 0 - don't know or definitely not trap
		# 1 - definitely trap
		self.KB = np.zeros(dims)
		self.maze = np.array(self.nSquares,dtype='str')

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
		initial_neighbor_KB = {action   : 0 for action in self.actions_name}
		self.set_position(((s_x,s_y),initial_neighbor_KB,0))
		self.maze[start] = 'S'

		goal = randoms[1]
		self.maze[goal] = 'O'

		self.maze = self.maze.reshape(dims)

		self.debug = 0

	def act(self, action):
		# state is now 3-tuple of (cur_xy, knowledge of each neighbor, "least trappiest quadrant")
		# if neighbor is known trap, then 1
		# if neighbor is unknown or not trap, then 0
		# "least trappiest quadrant" is quadrant (cur_xy as origin) with least traps
		xy, neighbor_KB, ltq = self.state
		x, y = xy

		new_x = x + self.movement[action][0]
		new_y = y + self.movement[action][1]

		action_name = self.actions_name[action]

		new_xy = (new_x,new_y)

		if(self.debug > 3):
			print(x,' ',y,' + ', self.actions_name[action], ' -> ',new_x, ' ', new_y)

		if(new_x < 0 or new_x >= self.xbound or new_y < 0 or new_y >= self.ybound):
			# Out of maze
			return -100

		if(self.maze[new_x][new_y] == 'X'):
			# Block
			self.KB[new_x][new_y] = 1
			neighbor_KB[action_name] = 1
			new_ltq = self.update_ltq(xy)
			self.state = (xy,neighbor_KB,new_ltq)

			return -100

		if(self.maze[new_x][new_y] == 'O'):
			# Target
			# Agent will now start a new random game
			return 100

		# now that we're moving, we need to update neighbors
		for i in range(self.actions):
			action_name = self.actions_name[i]
			neighbor_x = new_x + self.movement[i][0]
			neighbor_y = new_y + self.movement[i][1]

			try:
				neighbor_KB[action_name] = self.KB[neighbor_x][neighbor_y]
			except IndexError:
				# we're out of bounds
				neighbor_KB[action_name] = 1

		new_ltq = self.update_ltq(new_xy)
		self.state = (new_xy,neighbor_KB,new_ltq)

		# Punish for time passed
		return -0.5

	def update_ltq(self, xy):
		x, y = xy

		is_up = self.KB[:x,].sum() < self.KB[x+1:,].sum()
		is_left = self.KB[:,:y].sum() < self.KB[:,y+1:].sum()

		return 2 * is_up + is_left

	def observe(self):
		# The method will produce a single number representing distinct states
		xy, neighbor_KB, ltq = self.state
		x, y = xy
		sum = ltq * (2**4)
		for i in range(self.actions):
			action = self.actions_name[i]
			power = self.actions - i - 1
			sum += neighbor_KB[action] * (2**power)

		sum *= self.nSquares

		return int(sum + x*self.ybound + y)

	def reverse_observe(self,state_number):
		y = state_number % self.ybound
		state_number /= self.ybound

		x = state_number % self.xbound
		state_number /= self.xbound

		neighbor_KB = {}

		states = self.actions_name[:]
		states.reverse()

		for state in states:
			neighbor_KB.update({state   :   state_number % 2})
			state_number /= 2

		ltq = state_number % 4

		return ((x,y), neighbor_KB, ltq)

	def print_state(self, state):
		xy, neighbor_KB, ltq = state

		print(xy, end=' ')
		for kv in neighbor_KB.items():
			print(kv, end=' ')

		print(ltq, end=' ')

	def print_action(self, action):
		  print(self.actions_name[action], end=' ')

	def set_position(self,state):
		self.state = state
