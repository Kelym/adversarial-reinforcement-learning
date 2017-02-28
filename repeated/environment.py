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
        # ridiculous number of states
        self.states = self.nSquares * (3**self.nSquares)
        self.maze = np.array(self.nSquares,dtype='str')

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
        self.set_position(((s_x,s_y),{start :   ''}))
        self.maze[start] = 'S'

        goal = randoms[1]
        self.maze[goal] = 'O'

        self.maze = self.maze.reshape(dims)

    	self.actions = 4
    	self.movement = [[0,1],[1,0],[0,-1],[-1,0]]
    	self.actions_name = ['Right','Down','Left','Up']

    	self.debug = 0

    def act(self, action):
        # state is 2-tuple consisting of cur pos (x,y) and knowledge base (KB)
        # KB is dict whose keys are visited squares
        # and whose values are the maze contents at those squares
        xy, KB = self.state
        x, y = xy

    	new_x = x + self.movement[action][0]
    	new_y = y + self.movement[action][1]

        new_xy = (new_x,new_y)

    	if(self.debug > 3):
            print(x,' ',y,' + ', self.actions_name[action], ' -> ',new_x, ' ', new_y)

    	if(new_x < 0 or new_x >= self.xbound or new_y < 0 or new_y >= self.ybound):
            # Out of maze
            return -100

    	if(self.maze[new_x][new_y] == 'X'):
    	    # Block
            KB.update({new_xy   :   'X'})
            self.state = (xy,KB)
    	    return -100

    	if(self.maze[new_x][new_y] == 'O'):
            # Target
            # Agent will now start a new random game
            return 100

        KB.update({new_xy   :   ''})
        self.state = (new_xy,KB)
    	# Punish for time passed
    	return -0.5

    def observe(self):
        # The method will produce a single number representing distinct states
        xy, KB = self.state
        x, y = xy
        sum = 0
        count = 0
        for i in range(self.xbound):
            for j in range(self.ybound):
                count += 1
                power = self.nSquares - count
                coeff = 0
                try:
                    value = KB[(i,j)]
                    if value == '':
                        coeff = 1
                    else:
                        coeff = 2
                except KeyError:
                    pass

                sum += coeff * (3**power)

        sum *= self.nSquares

        return sum + x*self.ybound + y

    def reverse_observe(self,state_number):
        y = state_number % self.ybound
        state_number /= self.ybound

        x = state_number % self.xbound
        state_number /= self.xbound

        KB = {}

        xs = range(self.xbound)
        xs.reverse()
        ys = range(self.ybound)
        ys.reverse()

        for i in xs:
            for j in ys:
                coeff = state_number % 3
                state_number /= 3

                if coeff == 1:
                    KB.update({(i,j)    :   ''})
                elif coeff == 2:
                    KB.update({(i,j)    :   'X'})

        return ((x,y), KB)

    def print_state(self, state):
        xy, KB = state

        print(xy, end=' ')
        for kv in KB.items():
            print(kv, end=' ')

    def print_action(self, action):
      	print(self.actions_name[action], end=' ')

    def set_position(self,state):
        self.state = state
