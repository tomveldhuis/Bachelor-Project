from QLearning_Meta import *
from NeuralNetwork import *
from random import choices

def DeepQLearning(maze, parameters, verbose):
	nn = NeuralNetwork(10)
	
	memoryCapacity = 10
	sampleLength = 2
	
	nEpisodes = 1
	episodeLength = 10
	
	x = 0
	y = 0
	epsilon = 0.25
	
	goalReached = False
	for ep in range(nEpisodes):
		# Reset memory
		memory = []
		
		for count in range(episodeLength):
			# Get Q-values for current coordinates
			QValues = nn.predict(x, y)
			
			# Decide next action
			nextAction = epsilonGreedy(epsilon, QValues)
			
			# Get reward
			nextReward = reward(maze, x, y, nextAction)
			
			# Remember current state
			currentState = (x, y)
			
			# Go to the next state
			if nextAction == "north" and not maze.grid[y][x].walls[nextAction]:
				y -= 1
			if nextAction == "east" and not maze.grid[y][x].walls[nextAction]:
				x += 1
			if nextAction == "south" and not maze.grid[y][x].walls[nextAction]:
				y += 1
			if nextAction == "west" and not maze.grid[y][x].walls[nextAction]:
				x -= 1
			
			# Check if terminal state is reached
			if x == maze.size - 1 and y == maze.size - 1:
				goalReached = True
				break
			
			# Get Q-values from next state
			nextQValues = nn.predict(x, y)
			
			# Store the transition in memory
			nextState = (x, y)
			transition = (currentState, nextAction, nextReward, nextState)
			memory.append(transition)
			if len(memory) > memoryCapacity:
				memory.pop(0)
			
			# Retrieve a sample minibatch from memory
			minibatch = choices(memory, k=sampleLength)
		
		
	
