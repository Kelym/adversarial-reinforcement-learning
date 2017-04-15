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
    Return a single number representing the current state

The following is a naive environment that allows
  1) reading a rectangle shaped maze from file, where 'X' is block, 'O' is the reward place. 
  2) randomly generate a maze with given shape and number of traps. 
It will attempt to re-place the agent when it reaches the goal so the agent can play with the environment.
'''

class SameMazeMultiShot():
  def __init__(self):
    self.actions = 4
    self.movement = [[0,1],[1,0],[0,-1],[-1,0]]
    self.actions_name = ['Right','Down','Left','Up']
    self.debug = 0

  def read_maze(self,fname):
    with open(fname) as f:
      content = f.read().splitlines()

    self.maze = np.array([[1 if x=='O' else -1 if x=='X' else 0 for x in row] for row in content ])
    self.xbound = len(self.maze)
    self.ybound = len(self.maze[0])
    self.states = self.xbound * self.ybound
    # Search for a start symbol
    start = [x for x in content if 'S' in x]
    if(len(start)>0):
      start = start[0]
      self.set_position(self.maze.index(start), start.index('S'))
    else:
      if not self.random_start():
        print("The given maze does not have a valid start point")
    

  def generate_random_maze(self,x,y,trap):
    self.xbound = x
    self.ybound = y
    self.states = x * y
    while True:
      self.maze = np.zeros((i,j))
      candidates = np.random.choice(range(0, x * y), trap+1, replace=False)
      np.put(self.maze, candidates[0:-1], -1)
      self.maze[candidates[-1]] = 1
      if self.random_start():
        break


  def random_start(self):
    """Given a maze with target and obstacles, put the agent to a random starting point. Return true when succeeds and false if such a point
    cannot be found. 
    """
    (x,y) = np.where(self.maze == 1)
    init = (x[0],y[0])
    spfa_pending = [init]
    cur = 0
    visited = np.zeros((self.xbound, self.ybound))
    while cur < len(spfa_pending):
      x,y = spfa_pending[cur]
      cur += 1
      for move in self.movement:
        i = x+move[0]
        j = y+move[1]
        if (not self.out_of_maze(i,j) and 
                self.maze[i][j] == 0 and visited[i][j]==0):
          visited[i][j]=1
          spfa_pending.append((i,j))
    if cur == 1:
      return False 
    sel = np.random.randint(1, len(spfa_pending))
    self.set_position(spfa_pending[sel][0], spfa_pending[sel][1])
    return True


  def act(self, action):
    x = self.state[0]
    y = self.state[1]

    new_x = x + self.movement[action][0]
    new_y = y + self.movement[action][1]

    if(self.debug > 3):
      print(x,' ',y,' + ', self.actions_name[action], ' -> ',new_x, ' ', new_y)

    if self.out_of_maze(new_x, new_y) or self.is_trap(new_x, new_y):
      return -100
    self.set_position(new_x, new_y)
    if self.reach_target(new_x, new_y):
      # Target
      self.random_start() # Same maze, multiple shots
      return 100
    # Punish for time passed
    return -0.5

  def observe(self):
    # The method will produce a single number representing distinct states
    return self.state[0] * self.ybound + self.state[1];

  def print_state(self, state):
    print (state // self.ybound, state % self.ybound, end=' ')

  def print_action(self, action):
    print (self.actions_name[action], end=' ')

  def set_position(self,x,y):
    self.state = (x,y)

  def out_of_maze(self,x,y):
    return (x < 0 or x >= self.xbound or y < 0 or y >= self.ybound)

  def is_trap(self,x,y):
    return self.maze[x][y] < 0

  def reach_target(self,x,y):
    return self.maze[x][y] > 0
