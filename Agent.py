from __future__ import print_function
import numpy as np
import random
import Environment

class AgentQLearn():
  def __init__(self, env, curiosity, discount_rate = 0.9):
    self.env = env
    self.Q = [ [random.random()*0.5 for a in range(env.actions)] for s in range(env.states)]
    self.explore_chance = curiosity
    self.discount_rate = discount_rate
    
  def learn(self, learning_rate, steps):
    state = self.env.observe()
    for step in range(steps):

      if self.explore() :
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

      self.Q[state][action] = self.Q[state][action] + learning_rate*(reward + self.discount_rate*self.Q[new_state][next_action] - self.Q[state][action])
      
      state = new_state

  def explore(self):
    return random.random() < self.explore_chance

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
