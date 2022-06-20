from Maze import *
from QLearning import NormalQLearning, printNormalQTable
from DoubleQLearning import DoubleQLearning, printDoubleQTable

from pathlib import Path
from math import pow
import random
import matplotlib.pyplot as plt
import numpy as np
import sys

# Create a dictionary containing the used parameters
def parameters():
	dict = {
	# Learning rate
	# - "linear": 1 / number of states visited
	# - decimal number
	"alpha": "linear",
	# Discount factor
	"gamma": 0.95,
	# Epsilon (for epsilon-greedy exploration)
	"epsilon": 0.25,
	# Initial Q-value
	"initValue": 0,
	# Final epoch at which the Q-learning stops
	"finalEpoch": 10000,
	# Starting point
	"xStart": 0,
	"yStart": 0,
	# Use of a Q-table or neural network
	# - "QTable": a Q-table
	# - "neuralNetwork": a neural network
	"QForm": "QTable",
	# Type of TD algorithm used
	# - "target"
	# - "lambda"
	"TDForm": "target",
	# Lambda (for TD-lambda)
	"lambda": 0.5
	}
	return dict

def standardMazes():
	# Initialize array of mazes
	mazes = []
	random.seed(6122015)
	for i in range(3):
		mazes.append(Maze(5))
	for i in range(3):
		mazes.append(Maze(10))
	for i in range(3):
		mazes.append(Maze(20))
	
	# Reset the random seed
	random.seed()
	return mazes

# Perform an experiment in order to collect data
def experiment(parameters, mazes):
	n = 1000
	nMazes = len(mazes) 
	
	# Initialize results
	results = {}
	results["normal"] = {}
	results["double"] = {}
	
	xPoints = np.array(range(1, parameters["finalEpoch"]))
	
	Path("results").mkdir(parents=True, exist_ok=True)
	f = open("results/parameters.txt", "w")
	f.write("n: " + str(n) + "\n")
	f.write("nMazes: " + str(nMazes) + "\n")
	for x, y in parameters.items():
		f.write(str(x) + ": " + str(y) + "\n")
	f.close()
	
	sys.stdout.write("Running through the experiments...\n")
	for x in range(nMazes):
		results["normal"]["reward"] = []
		results["normal"]["maxQStart"] = []
		results["double"]["reward"] = []
		results["double"]["maxQStart"] = []
		
		maze = mazes[x]
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
		
		sys.stdout.write("Saving the collected data...\n")
		
		# Create maze files
		f = open("results/maze_" + str(x+1) + ".txt", "w")
		f.write(maze.asciiForm)
		f.close()
		
		# Create plot data
		f1 = open("results/reward_normal_" + str(x+1) + ".txt", "w")
		f2 = open("results/reward_double_" + str(x+1) + ".txt", "w")
		f3 = open("results/maxQStart_normal_" + str(x+1) + ".txt", "w")
		f4 = open("results/maxQStart_double_" + str(x+1) + ".txt", "w")
		for i in range(len(results["normal"]["reward"])):
			f1.write(str(results["normal"]["reward"][i]) + "\n")
			f2.write(str(results["double"]["reward"][i]) + "\n")
			f3.write(str(results["normal"]["maxQStart"][i]) + "\n")
			f4.write(str(results["double"]["maxQStart"][i]) + "\n")
		f1.close()
		f2.close()
		f3.close()
		f4.close()
		
		# Create plot files
		plt.figure()
		plt.plot(xPoints, results["normal"]["reward"], "r", label="Q-learning")
		plt.plot(xPoints, results["double"]["reward"], "b", label="Double Q-learning")
		plt.title("Maze: " + str(x+1) + "\nn = " + str(n) + ", finalEpoch = " + str(parameters["finalEpoch"]))
		plt.xlabel("Time steps")
		plt.ylabel("Average reward per time step")
		plt.legend(loc = "lower right")
		plt.savefig("results/reward_" + str(x+1) + ".png")
		
		plt.figure()
		plt.plot(xPoints, results["normal"]["maxQStart"], "r", label="Q-learning")
		plt.plot(xPoints, results["double"]["maxQStart"], "b", label="Double Q-learning")
		plt.title("Maze: " + str(x+1) + "\nn = " + str(n) + ", finalEpoch = " + str(parameters["finalEpoch"]))
		plt.xlabel("Time steps")
		plt.ylabel("max Q(S,a)")
		plt.legend(loc = "lower right")
		plt.savefig("results/maxQStart_" + str(x+1) + ".png")
		

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

def gridWorld():
	maze = Maze(3)
	
	txt = ""
	edge = "#######"
	nonEdge = "#     #"
	txt += edge + '\n'
	for i in range(5):
		txt += nonEdge + '\n'
	txt += edge
	
	maze.asciiForm = txt
	maze.grid = maze.convertToGrid(maze.asciiForm)
	maze.shortestPathLength = maze.determineShortestPathLength()
	return maze
