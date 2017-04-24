#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import random
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import Environment
import Agent

'''
learning_rate = 0.9
learning_step = 1000
discount_rate = 0.9
curiosity = 0.4
'''

trap_prob = 1.0 / 3
dimensions = (4,4)
record_file = 'out.txt'
model_file = 'model.pth'

random.seed(13)

# agent.print_policy((3,0))

def main():

	#sys.stdout = open(record_file, 'w')

	env = Environment.Toy(dimensions,trap_prob)
	env.debug = 1
	agent = Agent.AgentQLearn(env)

	results = agent.learn()
	agent.save_model(model_file)

	#results = ([1,2,3],[0.25,0.5,0.75])

	create_plots(results,'1-conv2d_1-linear_3000_GPU')

	#sys.stdout.close()

'''
	name = 'XOR_%d-%d-%d' % tuple(XOR_topology)
	experiment(XOR_topology, XOR_training, name)

	for nUnits_1 in range(1,5):
		for nUnits_2 in range(5):
			if nUnits_2 == 0:
				topology = [n,nUnits_1,4]
				name = 'multi_%d-%d-%d' % tuple(topology)
			else:
				topology = [n,nUnits_1,nUnits_2,4]
				name = 'multi_%d-%d-%d-%d' % tuple(topology)

			experiment(topology,training_set,name)

def experiment(topology,training_set,name):
	network = net.Network(topology)
	results = network.train(training_set, nEpochs, learning_rate)

	create_plots(results,name)

'''

def create_plots(results,name):
	folder = os.path.join(os.getcwd(),name)
	if not os.path.exists(folder):
		os.mkdir(folder)

	avg_rewards, success_rates = results
	epochs = range(len(avg_rewards))

	plot(avg_rewards, epochs, 'Avg. Rewards', 'Epochs', os.path.join(folder,'rewards.jpg'))
	plot(success_rates, epochs, 'Success Rates', 'Epochs', os.path.join(folder,'success_rates.jpg'))

def plot(y,x,yname,xname,fname):

	ys = [y]
	scatters = []

	colors = cm.rainbow(np.linspace(0, 1, len(ys)))
	for y, c in zip(ys, colors):
		scatters.append(plt.scatter(x, y, color=c, s=10))

	'''
	plt.legend(tuple(scatters),
		   (yname),
		   scatterpoints=1,
		   loc='lower right',
		   ncol=10,
		   fontsize=8)
	'''

	plt.title('%s vs. %s' % (yname,xname))
	#plt.show()
	plt.draw()
	fig = plt.gcf()
	fig.savefig(fname)
	plt.clf()

if __name__ == '__main__':
	main()