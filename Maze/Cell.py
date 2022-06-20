class Cell:
	def __init__(self, north, east, south, west):
		self.walls = {
		"north": bool(north),
		"east": bool(east),
		"south": bool(south),
		"west": bool(west)
		}
		# Variables for Q-learning
		self.traceValue = 0
		self.timesVisited = 0
		self.updateA = 0
		self.updateB = 0
		# Variables for calculating the shortest path length
		self.explored = False
		self.pathLength = 0
	
	def setCoordinates(self, x, y):
		self.x = x
		self.y = y
