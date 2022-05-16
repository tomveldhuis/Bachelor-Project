import sys
from QLearning_Meta import *

# (Normal) Q-learning algorithm
def NormalQLearning(maze, parameters, verbose):
	# Parameters
	size = maze.size
	epsilon = parameters["epsilon"]
	
	# Starting point
	x = parameters["xStart"]
	y = parameters["yStart"]
	
	if verbose:
		sys.stdout.write("--------\nQ-learning\n--------\n")
	
	# Initalize experimental data
	maze.initRewards()
	maze.initMaxQStart()
	
	count = 0
	goalReached = False
	while True:
		count += 1
		if verbose:
			sys.stdout.write("\rState: %i" % count)
		
		# Goal state is reached
		if x == size - 1 and y == size - 1 and not goalReached:
			goalReached = True
			if verbose:
				sys.stdout.write("\nGoal state reached\n")
		# Learning takes too long
		if count == parameters["finalEpoch"]:
			if verbose:
				sys.stdout.write(" (final epoch reached)\n")
			break
		
		# Determine the next action
		nextAction = epsilonGreedy(epsilon, maze.QValues[y][x])
		
		# Update the Q-value for the current state-action pair
		updateQValue(maze, x, y, parameters, count, nextAction)
		
		# Updating experimental data
		# - Average reward per time step
		maze.rewards.append(reward(maze, x, y, nextAction))
		# - Max Q-value of the starting state
		#maze.maxQStart.append(maze.QValues[maze.size - 2][maze.size - 1][maxAction(maze.QValues[maze.size - 2][maze.size - 1])])
		
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
def updateQValue(maze, x, y, parameters, count, nextAction):
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
	optimalFutureAction = maxAction(maze.QValues[futureY][futureX])
	
	# Determine TD (temporal difference) value
	TD = reward(maze, x, y, nextAction)
	TD += parameters["gamma"] * maze.QValues[futureY][futureX][optimalFutureAction]
	TD -= maze.QValues[y][x][nextAction]
	
	# Determine learning factor
	if parameters["alpha"] == "linear":
		learningFactor = 1 / count
	else:
		learningFactor = parameters["alpha"]
	
	# Update current Q-value
	maze.QValues[y][x][nextAction] += learningFactor * TD

# Prints the Q-table
def printNormalQTable(maze):
	sys.stdout.write("        ------------Q-values------------\n")
	sys.stdout.write("\tnorth\teast\tsouth\twest\n");
	for y in range(maze.size):
		for x in range(maze.size):
			sys.stdout.write("(%i, %i)\t" % (x, y))
			sys.stdout.write("%.2f\t" % maze.QValues[y][x]["north"])
			sys.stdout.write("%.2f\t" % maze.QValues[y][x]["east"])
			sys.stdout.write("%.2f\t" % maze.QValues[y][x]["south"])
			sys.stdout.write("%.2f\n" % maze.QValues[y][x]["west"])
