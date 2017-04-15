from __future__ import print_function
import numpy as np
import random
import Environment

class AgentQLearn():
  def __init__(self, env, curiosity, discount_rate = 0.9):
    self.env = env
    self.Q = np.random.uniform(0,0.5,(env.states, env.actions))
    self.explore_chance = curiosity
    self.discount_rate = discount_rate

  def choose_optimal_action(self, state):
    """Select the action to maximize the expected gain at the given state
    If multiple actions exist, randomly pick one.
    """
    actions = self.Q[state]
    optimal_gain = max(actions)
    optimal_actions = [i for i, j in enumerate(actions) if j == optimal_gain]
    return optimal_actions[np.random.randint(len(optimal_actions))]
    
  def learn(self, learning_rate, steps = 1000):
    """Do given steps of reinforcement Q learning
    """
    state = self.env.observe()
    for step in range(steps):
      if self.explore() :
        action = np.random.randint(self.env.actions)
      else:
        action = self.choose_optimal_action(state)

      reward = self.env.act(action)
      new_state = self.env.observe()

      if(self.env.debug > 2):
        self.env.print_state(state)
        self.env.print_action(action)
        print(reward, ' ', new_state)

      next_action = self.choose_optimal_action(new_state)

      self.Q[state][action] += learning_rate * (reward + self.discount_rate * self.Q[new_state][next_action] - self.Q[state][action])
      
      state = new_state

  def explore(self):
    return np.random.uniform() < self.explore_chance

  def print_policy(self):
    print(["{0:10}".format(i) for i in self.env.actions_name]) 
    for s in range(self.env.states):
      self.env.print_state(s)
      action = self.choose_optimal_action(s)
      self.env.print_action(action)
      print('')
      print(["{0:10.2f}".format(i) for i in self.Q[s]]) 

  def stimulate(self, steps = 10):
    s = self.env.observe()
    self.env.print_state(s)
    for i in range(0,steps):
      action = self.choose_optimal_action(s)
      self.env.print_action(action)
      reward = self.env.act(action)
      print("Gained ", reward, " => ", end=' ')
      s = self.env.observe()
      self.env.print_position(s)