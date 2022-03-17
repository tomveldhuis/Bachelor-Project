import sys
from Maze import *
from random import random, choice

# Returns the reward from a given state
def reward(size, x, y):
	if x == size - 1 and y == size - 1:
		return 100
	return 100 / (abs(x - size - 1) + abs(y - size - 1))

# Returns the possible actions-Q-value pairs for a given state
def possibleActions(maze, x, y):
	actions = maze.QValues[y][x].copy()
	for side, walled in maze.grid[y][x].walls.items():
		if walled:
			actions.pop(side)
	return actions

# Returns the action with the maximum Q-value for a given state
# and a given list of actions
def maxAction(maze, x, y, actions):
	maxValue = max([value for side, value in actions.items()])
	maxActions = [side for side, value in actions.items() if value == maxValue]
	if len(maxActions) > 1:
		return choice(maxActions)
	return maxActions[0]

# Returns an action for a given state (epsilon-greedy strategy)
def epsilonGreedy(maze, epsilon, x, y):
	actions = possibleActions(maze, x, y)
	# Exploitative action
	if random() > epsilon:
		return maxAction(maze, x, y, actions)
	# Explorative action
	return choice([side for side, value in actions.items()])

# Updates a Q-value for a given state-action pair
def updateQValue(maze, x, y, alpha, gamma, nextAction):
	# Determine future state
	futureX = x
	futureY = y
	if nextAction == "north":
		futureY -= 1
	if nextAction == "east":
		futureX += 1
	if nextAction == "south":
		futureY += 1
	if nextAction == "west":
		futureX -= 1
	# Determine optimal action for future state
	optimalFutureAction = maxAction(maze, futureX, futureY, possibleActions(maze, futureX, futureY))
	# Determine TD (temporal difference) value
	TD = reward(maze.size, x, y)
	TD += gamma * maze.QValues[futureY][futureX][optimalFutureAction]
	TD -= maze.QValues[y][x][nextAction]
	# Update current Q-value
	maze.QValues[y][x][nextAction] += alpha * TD

# Prints the Q-table
def printQTable(maze):
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
	size = maze.size
	alpha = 0.1
	gamma = 0.9
	epsilon = 0.25
	finalEpoch = 100000
	
	# Starting point
	x = 0
	y = 0
	
	count = 0
	goalReached = False
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
		nextAction = epsilonGreedy(maze, epsilon, x, y)
		
		# Update the Q-value for the current state-action pair
		updateQValue(maze, x, y, alpha, gamma, nextAction)
		
		# Go to the next state
		if nextAction == "north":
			y -= 1
		if nextAction == "east":
			x += 1
		if nextAction == "south":
			y += 1
		if nextAction == "west":
			x -= 1
