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

The following is a naive environment that reads a rectangle shaped maze from file, where 'X' is block, 'O' is the reward place.
'''

class Toy():
  def __init__(self, maze_file, start):
    self.maze = self.read_maze(maze_file)
    self.set_position(start)
    self.actions = 4
    self.movement = [[0,1],[1,0],[0,-1],[-1,0]]
    self.actions_name = ['Right','Down','Left','Up']
    self.xbound = len(self.maze)
    self.ybound = len(self.maze[0])
    self.states = self.xbound * self.ybound
    self.debug = 0

  def read_maze(self,fname):
    with open(fname) as f:
      content = f.read().splitlines()
    return content

  def act(self, action):

    x = self.state[0]
    y = self.state[1]

    new_x = x + self.movement[action][0]
    new_y = y + self.movement[action][1]

    if(self.debug > 3):
      print(x,' ',y,' + ', self.actions_name[action], ' -> ',new_x, ' ', new_y)

    if(new_x < 0 or new_x >= self.xbound or new_y < 0 or new_y >= self.ybound):
      # Out of maze
      return -100
    if(self.maze[new_x][new_y] == 'X'):
      # Block
      return -100

    self.state = (new_x, new_y)
    if(self.maze[new_x][new_y] == 'O'):
      # Target
      return 100
    # Punish for time passed
    return -0.5

  def observe(self):
    # The method will produce a single number representing distinct states
    return self.state[0]*self.ybound + self.state[1];

  def print_state(self, state):
    print(state/self.ybound, state%self.ybound, end=' ')

  def print_action(self, action):
    print(self.actions_name[action], end=' ')

  def set_position(self,state):
    self.state = state
