import sys
from QLearning_Meta import *

# Returns the possible actions-Q-value pairs for a given state
def possibleActions(maze, x, y):
	actions = maze.QValues[y][x].copy()
	for side, walled in maze.grid[y][x].walls.items():
		if walled:
			actions.pop(side)
	return actions

# Updates a Q-value for a given state-action pair
def updateQValue(maze, x, y, alpha, gamma, nextAction):
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
	TD += gamma * maze.QValues[futureY][futureX][optimalFutureAction]
	TD -= maze.QValues[y][x][nextAction]
	
	# Update current Q-value
	maze.QValues[y][x][nextAction] += alpha * TD

# Prints the Q-table
def printQTable(maze):
	sys.stdout.write("        ------------Q-values------------\n")
	sys.stdout.write("\tnorth\teast\tsouth\twest\n");
	for y in range(maze.size):
		for x in range(maze.size):
			sys.stdout.write("(%i, %i)\t" % (x, y))
			sys.stdout.write("%.2f\t" % maze.QValues[y][x]["north"])
			sys.stdout.write("%.2f\t" % maze.QValues[y][x]["east"])
			sys.stdout.write("%.2f\t" % maze.QValues[y][x]["south"])
			sys.stdout.write("%.2f\n" % maze.QValues[y][x]["west"])

# Q-learning algorithm
def QLearning(maze):
	# Parameters
	param = parameters()
	size = maze.size
	alpha = param["alpha"]
	gamma = param["gamma"]
	epsilon = param["epsilon"]
	initValue = param["initValue"]
	finalEpoch = param["finalEpoch"]
	
	# Starting point
	x = param["xStart"]
	y = param["yStart"]
	
	# Initialize Q-values
	maze.initQValues(initValue)
	
	count = 0
	goalReached = False
	sys.stdout.write("--------\nQ-learning\n--------\n")
	while True:
		count += 1
		sys.stdout.write("\rState: %i" % count)
		
		# Goal state is reached
		if x == size - 1 and y == size - 1 and not goalReached:
			goalReached = True
			sys.stdout.write("\nGoal state reached\n")
		# Learning takes too long
		if count == finalEpoch:
			sys.stdout.write(" (final epoch reached)\n")
			break
		
		# Determine the next action
		nextAction = epsilonGreedy(epsilon, maze.QValues[y][x])
		
		# Update the Q-value for the current state-action pair
		updateQValue(maze, x, y, alpha, gamma, nextAction)
		
		# Go to the next state
		if nextAction == "north" and not maze.grid[y][x].walls[nextAction]:
			y -= 1
		if nextAction == "east" and not maze.grid[y][x].walls[nextAction]:
			x += 1
		if nextAction == "south" and not maze.grid[y][x].walls[nextAction]:
			y += 1
		if nextAction == "west" and not maze.grid[y][x].walls[nextAction]:
			x -= 1
