from QLearning_Meta import *
from NeuralNetwork import *
from random import choices

def DeepQLearning(maze, parameters, verbose = False):
	nn = NeuralNetwork(10)
	
	nEpisodes = 1000
	episodeLength = 10
	
	x = 0
	y = 0
	
	goalReached = False
	for ep in range(nEpisodes):
		maze.grid[y][x].timesVisited += 1
		
		# Choose the prediction based on epsilon greedy
		currentActions = nn.getPrediction(x, y)
		
		# Perform the action from epsilon greedy
		nextAction = epsilonGreedyLinear(currentActions, maze.grid[y][x].timesVisited)
		
		# Get the new state and reward from the environment
		updateDeepQValue(maze, x, y, nn, parameters, nextAction)
		
		# Go to the next state
		if nextAction == "north" and not maze.grid[y][x].walls[nextAction]:
			y -= 1
		if nextAction == "east" and not maze.grid[y][x].walls[nextAction]:
			x += 1
		if nextAction == "south" and not maze.grid[y][x].walls[nextAction]:
			y += 1
		if nextAction == "west" and not maze.grid[y][x].walls[nextAction]:
			x -= 1
		

# Updates a Q-value for a given state-action pair
def updateDeepQValue(maze, x, y, nn, parameters, nextAction):
	# Determine future state
	futureX = x
	futureY = y
	if nextAction == "north" and not maze.grid[y][x].walls[nextAction]:
		futureY -= 1
	if nextAction == "east" and not maze.grid[y][x].walls[nextAction]:
		futureX += 1
	if nextAction == "south" and not maze.grid[y][x].walls[nextAction]:
		futureY += 1
	if nextAction == "west" and not maze.grid[y][x].walls[nextAction]:
		futureX -= 1
		
	# Determine optimal action for future state
	futureActions = nn.getPrediction(futureX, futureY)
	optimalFutureAction = maxAction(futureActions)
	
	# Determine TD (temporal difference) target value
	TD = reward(maze, x, y, nextAction)
	TD += parameters["gamma"] * futureActions[optimalFutureAction]
	
	# Update current Q-value
	nn.fitValue(x, y, TD)
		
		
	
