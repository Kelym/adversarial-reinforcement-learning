from __future__ import print_function
import numpy as np
import random

'''
The Environment Class shall all respond to two calls
  - act
    Do an action at the current state
    Move to the next state
    Return the reward
  - observe
    Return a single number representing the current state

The following is a naive environment that allows
  1) reading a rectangle shaped maze from file, where 'X' is block, 'O' is the reward place, (optionally) 'S' specifies a start point.
  2) randomly generate a maze with given shape and number of traps. 
It will attempt to re-place the agent and (optionally) refresh the maze when the agent reaches the goal, so the agent can play with the environment.
'''

class Toy():
  def __init__(self):
    self.actions = 4
    self.movement = [[0,1],[1,0],[0,-1],[-1,0]]
    self.actions_name = ['Right','Down','Left','Up']
    self.debug = 0

  def read_maze(self,fname):
    content = open(fname).read().splitlines()
    self.maze = np.array([[1 if x=='O' else -1 if x=='X' else 0
                          for x in row] for row in content ])
    self.maze_id = Toy.generate_maze_id(self.maze)
    self.maze_pretty = content
    self.xbound = len(self.maze)
    self.ybound = len(self.maze[0])
    self.states = (2**(self.xbound * self.ybound) * 
                    (self.xbound * self.ybound)**2)
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
    self.states = 2**(self.xbound * self.ybound) * self.xbound * self.ybound
    while True:
      self.maze = np.zeros((i,j))
      candidates = np.random.choice(range(0, x * y), trap+1, replace=False)
      np.put(self.maze, candidates[0:-1], -1)
      self.maze[candidates[-1]] = 1
      if self.random_start():
        self.maze_id = Toy.generate_maze_id(self.maze)
        self.maze_pretty = [''.join(['X' if x==-1 else '.' if x==0 else 1
                            for x in row]) for row in self.maze]
        break

  @staticmethod
  def generate_maze_id(maze):
    (sx,sy) = maze.shape
    blocks = sum(1<<i for i, b in enumerate(maze.flatten()) if b == -1)
    (x,y) = np.where(maze == 1)
    target_pos = x[0]*sy+y[0]
    return ((target_pos * (1<<(sx*sy)) + blocks) * 
            (sx*sy))

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
    x = self.cur[0]
    y = self.cur[1]

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
    pos = self.cur[0] * self.ybound + self.cur[1]
    return pos + self.maze_id

  def print_position(self, state):
    cur = state % (self.xbound * self.ybound)
    print (cur // self.ybound, cur % self.ybound)

  def print_maze(self):
    print('\n'.join(self.maze_pretty))

  def print_state(self, state):
    self.print_maze()
    self.print_position(state)

  def print_action(self, action):
    print (self.actions_name[action], end=' ')

  def set_position(self,x,y):
    self.cur = (x,y)

  def out_of_maze(self,x,y):
    return (x < 0 or x >= self.xbound or y < 0 or y >= self.ybound)

  def is_trap(self,x,y):
    return self.maze[x][y] < 0

  def reach_target(self,x,y):
    return self.maze[x][y] > 0
