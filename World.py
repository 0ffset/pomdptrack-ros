class World:
	def __init__(self, grid = None):
		self.grid = World.__getDefaultGrid() if grid == None else grid
		self.gridNumRows = len(self.grid)
		self.gridNumCols = len(self.grid[0])

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
