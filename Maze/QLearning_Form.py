from Maze import *
from QLearning import NormalQLearning, printNormalQTable
from DoubleQLearning import DoubleQLearning, printDoubleQTable

import matplotlib.pyplot as plt
import numpy as np
import sys

# Create a dictionary containing the used parameters
def parameters():
	dict = {
	# Learning rate
	"alpha": 0.1,
	# Discount factor
	"gamma": 0.9,
	# Epsilon (for epsilon-greedy exploration)
	"epsilon": 0.25,
	# Initial Q-value
	"initValue": 0,
	# Final epoch at which the Q-learning stops
	"finalEpoch": 1000,
	# Starting point
	"xStart": 0,
	"yStart": 0
	}
	return dict

# Perform an experiment in order to collect data
def experiment(parameters):
	size = 6
	n = 1000
	nMazes = 1
	
	# Initialize results
	results = {}
	results["normal"] = {}
	results["double"] = {}
	
	xPoints = np.array(range(1, parameters["finalEpoch"]))
	
	sys.stdout.write("Running through the experiments...\n")
	for x in range(nMazes):
		results["normal"]["reward"] = []
		results["normal"]["maxQStart"] = []
		results["double"]["reward"] = []
		results["double"]["maxQStart"] = []
		
		maze = Maze(size)
		sys.stdout.write("Randomly generated maze: %i (out of %i)\n" % (x+1, nMazes))
		for i in range(n):
			sys.stdout.write("\rExperiment: %i (out of %i)" % (i+1, n))
			
			# (Normal) Q-learning
			QLearning(maze, "normal")
			if i == 0:
				for j in range(parameters["finalEpoch"] - 1):
					results["normal"]["reward"].append(maze.rewards[j])
					results["normal"]["maxQStart"].append(maze.maxQStart[j])
			else:
				for j in range(parameters["finalEpoch"] - 1):
					results["normal"]["reward"][j] += maze.rewards[j]
					results["normal"]["maxQStart"][j] += maze.maxQStart[j]
			
			# Double Q-learning
			QLearning(maze, "double")
			if i == 0:
				for j in range(parameters["finalEpoch"] - 1):
					results["double"]["reward"].append(maze.rewards[j])
					results["double"]["maxQStart"].append(maze.maxQStart[j])
			else:
				for j in range(parameters["finalEpoch"] - 1):
					results["double"]["reward"][j] += maze.rewards[j]
					results["double"]["maxQStart"][j] += maze.maxQStart[j]

		# Average the data
		sys.stdout.write("\nAveraging the collected data...\n")
		for i in range(parameters["finalEpoch"] - 1):
			results["normal"]["reward"][i] /= n
			results["normal"]["maxQStart"][i] /= n
			results["double"]["reward"][i] /= n
			results["double"]["maxQStart"][i] /= n
		
		# Create plot data
		plt.figure()
		plt.plot(xPoints, results["normal"]["reward"], "r")
		plt.plot(xPoints, results["double"]["reward"], "b")
		plt.title("Maze: " + str(x+1) + "\nn = " + str(n) + ", finalEpoch = " + str(parameters["finalEpoch"]))
		plt.xlabel("Time steps")
		plt.ylabel("Average reward per time step")
		
		plt.figure()
		plt.plot(xPoints, results["normal"]["maxQStart"], "r")
		plt.plot(xPoints, results["double"]["maxQStart"], "b")
		plt.title("Maze: " + str(x+1) + "\nn = " + str(n) + ", finalEpoch = " + str(parameters["finalEpoch"]))
		plt.xlabel("Time steps")
		plt.ylabel("max Q(S,a)")
	
	# Plot the data
	sys.stdout.write("Printing the plots...\n")
	plt.show()

def QLearning(maze, form = "normal", newQTable = True, verbose = False):
	if form == "normal":
		if newQTable:
			maze.initQValues(parameters()["initValue"])
		NormalQLearning(maze, parameters(), verbose)
	elif form == "double":
		if newQTable:
			maze.initDoubleQValues(parameters()["initValue"])
		DoubleQLearning(maze, parameters(), verbose)

def printQTable(maze, form, newQTable = True):
	if form == "normal":
		printNormalQTable(maze)
	elif form == "double":
		printDoubleQTable(maze)
