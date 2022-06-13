import sys
from QLearning_Meta import *

# Double Q-learning algorithm
def DoubleQLearning(maze, parameters, verbose):
	# Parameters
	size = maze.size
	alpha = parameters["alpha"]
	gamma = parameters["gamma"]
	epsilon = parameters["epsilon"]
	finalEpoch = parameters["finalEpoch"]
	
	# Starting point
	x = parameters["xStart"]
	y = parameters["yStart"]
	
	if verbose:
		sys.stdout.write("--------\nDouble Q-learning\n--------\n")
	
	# Initialize experimental data
	maze.initRewards()
	
	count = 0
	goalReached = False
	while True:
		count += 1
		maze.grid[y][x].timesVisited += 1
		
		if verbose:
			sys.stdout.write("\rState: %i" % count)
		
		# Goal state is reached
		if x == size - 1 and y == size - 1 and not goalReached:
			goalReached = True
			if verbose:
				sys.stdout.write("\nGoal state reached\n")
		# Learning takes too long
		if count == finalEpoch:
			if verbose:
				sys.stdout.write(" (final epoch reached)\n")
			break
		
		# Determine the next action
		nextAction = epsilonGreedyLinear(averageActions(maze, x, y), maze.grid[y][x].timesVisited)
		
		# Update the Q-value for the current state-action pair
		updateDoubleQValue(maze, x, y, parameters, count, nextAction)
		
		# Update eligibility traces
		if parameters["TDForm"] == "lambda":
			maze.updateTraces(x, y, parameters["gamma"], parameters["lambda"])
		
		# Updating experimental data
		# - Average reward per time step
		maze.rewards.append(reward(maze, x, y, nextAction))
		# - Max Q-value of the starting state
		maze.maxQStart.append(averageActions(maze, maze.size - 2, maze.size - 1)[maxAction(averageActions(maze, maze.size - 2, maze.size - 1))])
		
		# Go to the next state
		if nextAction == "north" and not maze.grid[y][x].walls[nextAction]:
			y -= 1
		if nextAction == "east" and not maze.grid[y][x].walls[nextAction]:
			x += 1
		if nextAction == "south" and not maze.grid[y][x].walls[nextAction]:
			y += 1
		if nextAction == "west" and not maze.grid[y][x].walls[nextAction]:
			x -= 1

# Returns the average Q-value over the double Q-values
# (both corresponding to a given state)
def averageActions(maze, x, y):
	actionsA = maze.QValues[0][y][x]
	actionsB = maze.QValues[1][y][x]
	actions = {}
	for side, walled in maze.grid[y][x].walls.items():
		average = (actionsA[side] + actionsB[side]) / 2
		actions[side] = average
	return actions
	
# Updates a Q-value for a given state-action pair
# (random choice between double Q-values)
def updateDoubleQValue(maze, x, y, parameters, count, nextAction):
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
	
	# Determine which Q-value to change
	choiceQ = choice([0, 1])
	
	# Determine optimal action for future state
	optimalFutureAction = maxAction(maze.QValues[choiceQ][futureY][futureX])
	
	# Determine TD (temporal difference) value
	TD = reward(maze, x, y, nextAction)
	TD += parameters["gamma"] * maze.QValues[1 - choiceQ][futureY][futureX][optimalFutureAction]
	TD -= maze.QValues[choiceQ][y][x][nextAction]
	
	# Determine learning factor
	if parameters["alpha"] == "linear":
		learningFactor = 1 / count
	else:
		learningFactor = parameters["alpha"]
	
	# Update current Q-value
	TD *= learningFactor
	if parameters["TDForm"] == "lambda":
		TD *= maze.grid[y][x].traceValue
	maze.QValues[choiceQ][y][x][nextAction] += TD

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
