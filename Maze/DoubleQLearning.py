import sys
from QLearning_Meta import *

# Returns the possible actions-Q-value pairs for a given state
# (one of the two possible Q-values set)
def possibleActionsSelect(maze, x, y, select):
	actions = maze.QValues[select][y][x].copy()
	for side, walled in maze.grid[y][x].walls.items():
		if walled:
			actions.pop(side)
	return actions

# Returns the possible actions-Q-value pairs for a given state
# (average of double Q-values)
def possibleAverageActions(maze, x, y):
	actionsA = maze.QValues[0][y][x]
	actionsB = maze.QValues[1][y][x]
	averageActions = {}
	for side, walled in maze.grid[y][x].walls.items():
		if not walled:
			average = (actionsA[side] + actionsB[side]) / 2
			averageActions[side] = average
	return averageActions
	
# Updates a Q-value for a given state-action pair
# (random choice between double Q-values)
def updateDoubleQValue(maze, x, y, alpha, gamma, nextAction):
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
	
	# Determine which Q-value to change
	choiceQ = choice([0, 1])
	
	# Determine optimal action for future state
	optimalFutureAction = maxAction(maze, futureX, futureY, possibleActionsSelect(maze, futureX, futureY, choiceQ))
	
	# Determine TD (temporal difference) value
	TD = reward(maze.size, x, y)
	TD += gamma * maze.QValues[1 - choiceQ][futureY][futureX][optimalFutureAction]
	TD -= maze.QValues[choiceQ][y][x][nextAction]
	
	# Update current Q-value
	maze.QValues[choiceQ][y][x][nextAction] += alpha * TD

# Prints the double Q-table
def printDoubleQTable(maze):
	sys.stdout.write("        ----------Q-values (A)----------|----------Q-values (B)----------\n")
	sys.stdout.write("\tnorth\teast\tsouth\twest\t|north\teast\tsouth\twest\n");
	for y in range(maze.size):
		for x in range(maze.size):
			sys.stdout.write("(%i, %i)\t" % (x, y))
			sys.stdout.write("%.2f\t" % maze.QValues[0][y][x]["north"])
			sys.stdout.write("%.2f\t" % maze.QValues[0][y][x]["east"])
			sys.stdout.write("%.2f\t" % maze.QValues[0][y][x]["south"])
			sys.stdout.write("%.2f\t|" % maze.QValues[0][y][x]["west"])
			sys.stdout.write("%.2f\t" % maze.QValues[1][y][x]["north"])
			sys.stdout.write("%.2f\t" % maze.QValues[1][y][x]["east"])
			sys.stdout.write("%.2f\t" % maze.QValues[1][y][x]["south"])
			sys.stdout.write("%.2f\t\n" % maze.QValues[1][y][x]["west"])

# Double Q-learning algorithm
def DoubleQLearning(maze):
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
	
	# Initialize double Q-values
	maze.initDoubleQValues(initValue)
	
	count = 0
	goalReached = False
	sys.stdout.write("--------\nDouble Q-learning\n--------\n")
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
		nextAction = epsilonGreedy(maze, epsilon, x, y, possibleAverageActions(maze, x, y))
		
		# Update the Q-value for the current state-action pair
		updateDoubleQValue(maze, x, y, alpha, gamma, nextAction)
		
		# Go to the next state
		if nextAction == "north":
			y -= 1
		if nextAction == "east":
			x += 1
		if nextAction == "south":
			y += 1
		if nextAction == "west":
			x -= 1
