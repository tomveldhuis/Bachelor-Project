from Cell import *
from NeuralNetwork import *
from random import shuffle, randrange

class Maze:
	# Constructor
	def __init__(self, size):
		self.size = int(size)
		self.asciiForm = self.makeRandomMaze(self.size, self.size)
		self.grid = self.convertToGrid(self.asciiForm)
		self.shortestPathLength = self.determineShortestPathLength()
	
	# Generates a random ASCII maze
	def makeRandomMaze(self, w, h):
		# From: https://rosettacode.org/wiki/Maze_generation#Python
		vis = [[0] * w + [1] for _ in range(h)] + [[1] * (w + 1)]
		ver = [["# "] * w + ['#'] for _ in range(h)] + [[]]
		hor = [["##"] * w + ['#'] for _ in range(h + 1)]
	 
		def walk(x, y):
			vis[y][x] = 1
	 
			d = [(x - 1, y), (x, y + 1), (x + 1, y), (x, y - 1)]
			shuffle(d)
			for (xx, yy) in d:
				if vis[yy][xx]: continue
				if xx == x: hor[max(y, yy)][x] = "# "
				if yy == y: ver[y][max(x, xx)] = "  "
				walk(xx, yy)
	 
		walk(randrange(w), randrange(h))
	 
		s = ""
		for (a, b) in zip(hor, ver):
			s += ''.join(a + ['\n'] + b + ['\n'])
		return s[:-2]
	
	# Converts an ASCII maze to a grid with cells
	def convertToGrid(self, asciiForm):
		# Create a list
		gridList = []
		width = 2 * self.size + 1
		for y in range(width):
			start = y * width + y
			end = (y + 1) * width + y
			gridList.append(asciiForm[start:end])
		
		# Create a grid
		grid = []
		for y in range(width):
			if y % 2 == 1:
				line = []
				for x in range(width):
					if x % 2 == 1:
						north = False
						if gridList[y-1][x] == "#":
							north = True
						east = False
						if gridList[y][x+1] == "#":
							east = True
						south = False
						if gridList[y+1][x] == "#":
							south = True
						west = False
						if gridList[y][x-1] == "#":
							west = True
						line.append(Cell(north, east, south, west))
				grid.append(line)
		
		for y in range(len(grid)):
			for x in range(len(grid[y])):
				grid[y][x].setCoordinates(x, y)
		
		return grid
	
	# Determines length of the shortest path
	def determineShortestPathLength(self):
		queue = []
		self.grid[0][0].explored = True
		queue.append(self.grid[0][0])
		
		while len(queue) != 0:
			# Check
			cell = queue.pop(0)
			#print(str(cell.x) + " " + str(cell.y))
			if cell.x == self.size - 1 and cell.y == self.size - 1:
				return cell.pathLength
				
			# Search
			actions = [x for x, y in cell.walls.items() if y == False]
			#print(actions)
			for side in actions:
				x = cell.x
				y = cell.y
				if side == "north":
					y -= 1
				if side == "east":
					x += 1
				if side == "south":
					y += 1
				if side == "west":
					x -= 1
				
				if not self.grid[y][x].explored:
					self.grid[y][x].explored = True
					self.grid[y][x].pathLength = cell.pathLength + 1
					queue.append(self.grid[y][x])
			
	
	#---------------- Q-learning ----------------------
	
	# Creates a new grid of Q-values for the maze
	def newQValues(self, initValue):
		dict = {
		"north": float(initValue),
		"east": float(initValue),
		"south": float(initValue),
		"west": float(initValue)
		}
		QValues = []
		for y in range(self.size):
			line = []
			for x in range(self.size):
				line.append(dict.copy())
			QValues.append(line)
		return QValues
	
	# Initialize single Q-values for the maze
	def initQValues(self, initValue):
		self.QValues = self.newQValues(initValue)
	
	# Initialize double Q-values for the maze
	def initDoubleQValues(self, initValue):
		QValues = []
		QValues.append(self.newQValues(initValue))
		QValues.append(self.newQValues(initValue))
		self.QValues = QValues
	
	# Update eligibility traces (TD(lambda) training)
	def updateTraces(self, visitX, visitY, gamma, Lambda):
		for y in range(self.size):
			for x in range(self.size):
				self.grid[y][x].traceValue *= gamma * Lambda
				if x == visitX and y == visitY:
					self.grid[y][x].traceValue += 1
					
	# 
	def initNeuralNetwork(self, initValue):
		self.neuralNetwork = NeuralNetwork(self.size)
	
	#----------------- Experimental data -----------------
	
	# Initialize list of rewards (for experiments)
	def initRewards(self):
		self.rewards = []
	
	# Initialize list of max Q-values of starting state (for experiments)
	def initMaxQStart(self):
		self.maxQStart = []
		
