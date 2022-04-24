class Cell:
	def __init__(self, north, east, south, west):
		self.walls = {
		"north": bool(north),
		"east": bool(east),
		"south": bool(south),
		"west": bool(west)
		}
		self.explored = False
		self.pathLength = 0
	
	def setCoordinates(self, x, y):
		self.x = x
		self.y = y
