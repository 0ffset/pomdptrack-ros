import random

class World:
	# States keywords
	UPPER_LEFT   = 0
	UPPER_RIGHT  = 1
	BOTTOM_LEFT  = 2
	BOTTOM_RIGHT = 3
	RANDOM_STATE = 4

	def __init__(self, grid = None):
		self.grid           = World.__getDefaultGrid() if grid == None else grid
		self.gridNumRows    = len(self.grid)
		self.gridNumCols    = len(self.grid[0])
		self.robotStates    = []
		self.__initRobotStates()
		self.numRobotStates = len(self.robotStates)

	@staticmethod
	def __getDefaultGrid():
		"""Returns a default grid."""
		return [[0, 0, 0, 0, 0],
				[0, 0, 1, 0, 0],
				[0, 0, 1, 0, 0],
				[0, 0, 1, 0, 0],
				[0, 0, 0, 0, 0]]

	def isWall(self, cell):
		"""Determines whether a cell is a wall in the grid."""
		i, j = cell[0], cell[1]
		return self.grid[i][j] == 1

	def isWithinBounds(self, cell):
		"""Determines whether a cell is within grid bounds."""
		i, j = cell[0], cell[1]
		return i >= 0 and i < self.gridNumRows and j >= 0 and j < self.gridNumCols

	def isValidRobotState(self, cell):
		"""Determines if a cell is a valid robot state."""
		return self.isWithinBounds(cell) and not self.isWall(cell)

	def getNeighboringRobotStates(self, cell):
		"""Returns a set of cells in the neighborhood of cell."""
		relNeighborCoords = [(-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1)]
		neighbors = set()
		for v in relNeighborCoords:
			neighbor = (cell[0] + v[0], cell[1] + v[1])
			if self.isValidRobotState(neighbor):
				neighbors.add(neighbor)
		return neighbors

	def __initRobotStates(self):
		"""Enumerate all valid robot states in grid."""
		for i in range(self.gridNumRows):
			for j in range(self.gridNumCols):
				state = (i, j)
				if self.isValidRobotState(state):
					self.robotStates.append(state)

	def getState(self, keyword):
		"""Returns the state corresponding to given keyword."""
		if keyword == World.UPPER_LEFT:
			return (0, 0)
		elif keyword == World.UPPER_RIGHT:
			return (0, self.gridNumCols - 1)
		elif keyword == World.BOTTOM_LEFT:
			return (self.gridNumRows - 1, 0)
		elif keyword == World.BOTTOM_RIGHT:
			return (self.gridNumRows - 1, self.gridNumCols - 1)
		elif keyword == World.RANDOM_STATE:
			i = random.randint(0, self.numRobotStates - 1)
			return self.robotStates[i]

	def printWorld(self, state):
		"""Prints world based on Model state."""
		agentStates, targetStates = state[0], state[1]
		for i in range(self.gridNumRows):
			row = []
			for j in range(self.gridNumCols):
				if self.grid[i][j] == 1:
					row.append("#")
				elif (i, j) in agentStates and (i, j) in targetStates:
					row.append("@")
				elif (i, j) in agentStates:
					row.append("a")
				elif (i, j) in targetStates:
					if len(targetStates) > 1 and targetStates[0] == (i, j) and targetStates[1] == (i, j):
						row.append("T")
					else:
						for k in range(len(targetStates)):
							if targetStates[k] == (i, j):
								row.append(str(k + 1))
				else:
					row.append(".")
			print " ".join(row)
