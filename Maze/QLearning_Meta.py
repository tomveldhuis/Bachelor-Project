from Maze import *
from random import random, choice

# Returns the reward from a given state
def reward(maze, x, y, action):
	# ---For grid world experiments---
	#return rewardGridWorld(maze, x, y, action)
	
	# Enter the goal state from the west (if possible)
	if x == maze.size - 2 and y == maze.size - 1:
		if action == "east" and not maze.grid[y][x].walls[action]:
			return 100
			
	# Enter the goal state from the north (if possible)
	if x == maze.size - 1 and y == maze.size - 2:
		if action == "south" and not maze.grid[y][x].walls[action]:
			return 100
			
	# Hit a wall
	if maze.grid[y][x].walls[action] == True:
		return -100
	return 0
	
# Returns the action with the maximum Q-value for a given state
# and a given list of actions
def maxAction(actions):
	maxValue = max([value for side, value in actions.items()])
	maxActions = [side for side, value in actions.items() if value == maxValue]
	if len(maxActions) > 1:
		return choice(maxActions)
	return maxActions[0]

# Returns an action for a given state (epsilon-greedy strategy)
def epsilonGreedy(epsilon, actions):
	# Exploitative action
	if random() > epsilon:
		return maxAction(actions)
	# Explorative action
	return choice([side for side, value in actions.items()])

def rewardGridWorld(maze, x, y, action):
	if x == maze.size - 1 and y == maze.size - 1:
		return 5
	return choice([-12, 10])
