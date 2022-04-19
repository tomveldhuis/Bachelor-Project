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
	"finalEpoch": 10000,
	# Starting point
	"xStart": 0,
	"yStart": 0
	}
	return dict

# Perform an experiment in order to collect data
def experiment(parameters):
	size = 10
	n = 100
	nMazes = 3
	
	results = {}
	results["normal"] = {}
	results["double"] = {}
	results["normal"]["reward"] = []
	results["normal"]["maxQStart"] = []
	results["double"]["reward"] = []
	results["double"]["maxQStart"] = []
	
	
	sys.stdout.write("Running through the experiments...\n")
	for x in range(nMazes):
		normalReward = []
		normalMaxQ = []
		doubleReward = []
		doubleMaxQ = []
		
		maze = Maze(size)
		sys.stdout.write("Randomly generated maze: %i (out of %i)\n" % (x+1, nMazes))
		for i in range(n):
			sys.stdout.write("\rExperiment: %i (out of %i)" % (i+1, n))
			
			# (Normal) Q-learning
			QLearning(maze, "normal")
			if i == 0:
				for j in range(parameters["finalEpoch"] - 1):
					normalReward.append(maze.rewards[j])
					normalMaxQ.append(maze.maxQStart[j])
			else:
				for j in range(parameters["finalEpoch"] - 1):
					normalReward[j] += maze.rewards[j]
					normalMaxQ[j] += maze.maxQStart[j]
			
			# Double Q-learning
			QLearning(maze, "double")
			if i == 0:
				for j in range(parameters["finalEpoch"] - 1):
					doubleReward.append(maze.rewards[j])
					doubleMaxQ.append(maze.maxQStart[j])
			else:
				for j in range(parameters["finalEpoch"] - 1):
					doubleReward[j] += maze.rewards[j]
					doubleMaxQ[j] += maze.maxQStart[j]

		# Average the data and add it to the total results
		sys.stdout.write("\nAveraging the collected data...\n")
		for i in range(parameters["finalEpoch"] - 1):
			normalReward[i] /= n
			normalMaxQ[i] /= n
			doubleReward[i] /= n
			doubleMaxQ[i] /= n
			
			if x == 0:
				results["normal"]["reward"].append(normalReward[i])
				results["normal"]["maxQStart"].append(normalMaxQ[i])
				results["double"]["reward"].append(doubleReward[i])
				results["double"]["maxQStart"].append(doubleMaxQ[i])
			else:
				results["normal"]["reward"][i] += normalReward[i]
				results["normal"]["maxQStart"][i] += normalMaxQ[i]
				results["double"]["reward"][i] += doubleReward[i]
				results["double"]["maxQStart"][i] += doubleMaxQ[i]
	
	# Average total results
	sys.stdout.write("Getting the total results...\n")
	for i in range(parameters["finalEpoch"] - 1):
		results["normal"]["reward"][i] /= nMazes
		results["normal"]["maxQStart"][i] /= nMazes
		results["double"]["reward"][i] /= nMazes
		results["double"]["maxQStart"][i] /= nMazes
	
	# Plot the data
	sys.stdout.write("Printing the plots...\n")
	xPoints = np.array(range(1, parameters["finalEpoch"]))
	
	plt.plot(xPoints, results["normal"]["reward"], "r")
	plt.plot(xPoints, results["double"]["reward"], "b")
	plt.title("n = " + str(n) + ", nMazes = " + str(nMazes) + ", finalEpoch = " + str(parameters["finalEpoch"]))
	plt.xlabel("Time steps")
	plt.ylabel("Average reward per time step")
	plt.figure()
	
	plt.plot(xPoints, results["normal"]["maxQStart"], "r")
	plt.plot(xPoints, results["double"]["maxQStart"], "b")
	plt.title("n = " + str(n) + ", nMazes = " + str(nMazes) + ", finalEpoch = " + str(parameters["finalEpoch"]))
	plt.xlabel("Time steps")
	plt.ylabel("max Q(S,a)")
	
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
