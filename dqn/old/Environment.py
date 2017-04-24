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
		start = int(randoms[0])
		s_x = int(start / self.ybound)
		s_y = start % self.ybound

		# initial starting position
		self.pos = (s_x, s_y)

		self.maze[start] = 'S'

		goal = randoms[1]
		self.maze[goal] = 'G'

		self.maze = self.maze.reshape(dims)

		self.debug = 0

	def dims(self):
		return self.xbound, self.ybound

	def new_maze(self):
		return Toy((self.xbound, self.ybound), self.trap_prob)

	def act(self, action):

		x,y = self.pos

		new_x = int(x + self.movement[action][0])
		new_y = int(y + self.movement[action][1])

		if(self.debug > 3):
			print(x,' ',y,' + ', self.actions_name[action], ' -> ',new_x, ' ', new_y)

		if(new_x < 0 or new_x >= self.xbound or new_y < 0 or new_y >= self.ybound):
			# Out of maze; considered a "trap"
			return -100, 'T'
		if(self.maze[new_x][new_y] == 'T'):
			# Trap
			return -100, 'T'

		self.set_position((new_x, new_y))
		if(self.maze[new_x][new_y] == 'G'):
			# Target
			return 100, 'G'
		# Punish for time passed
		return -0.5, 'P'

	def observe(self):
		# The method will produce a 3D array (self.dims x 3 flags)
		# the three flags are as follows:
		#	- 1 if square is current location of player
		#	- 1 if square is trap
		#	- 1 if square is goal 
		x, y = self.maze.shape
		state_pos = np.zeros((x,y),dtype=int)
		state_trap = np.zeros((x,y),dtype=int)
		state_goal = np.zeros((x,y),dtype=int)
		for i in range(self.xbound):
			for j in range(self.ybound):
				square = self.maze[i,j]
				
				state_pos[i,j] = int(self.pos == (i,j))
				state_trap[i,j] = int(self.isTrap(square))
				state_goal[i,j] = int(self.isGoal(square))

		return np.array([state_pos,state_trap,state_goal])

	def print_state(self, state=None):
		state_display = None

		# print current state of environment
		if state is None:
			state_display = self.maze.copy()
			x,y = self.pos
			square = self.maze[x,y]
			if self.isTrap(square):
				state_display[x,y] = 'P/T'
			elif self.isGoal(square):
				state_display[x,y] = 'P/G'
			elif self._isStart(square):
				state_display[x,y] = 'P/S'
			else:
				state_display[x,y] = 'P'

		else:
			state_display = np.zeros(self.maze.shape,dtype=str)
			pos, trap, goal = tuple(state)
			for i in range(self.xbound):
				for j in range(self.ybound):
				
					if trap[i,j]:
						state_display[i,j] = 'P/T' if pos[i,j] else 'T'
	
					if goal[i,j]:
						state_display[i,j] = 'P/G' if pos[i,j] else 'G'
	
					if pos[i,j]:
						state_display[i,j] = 'P'
				
		for row in state_display:
			print(row)

		return state_display

	def isTrap(self,square):
		return square == 'T'

	def isGoal(self,square):
		return square == 'G'

	def _isStart(self,square):
		return square == 'S'

	def print_action(self, action):
		print(self.actions_name[action], end=' ')

	def set_position(self,pos):
		self.pos = pos
