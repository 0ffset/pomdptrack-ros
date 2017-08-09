from Model import Model
import time

if __name__ == "__main__":
	#grid = [[0, 0, 0, 0, 0, 0, 0, 0],
	#		[0, 0, 0, 0, 1, 0, 0, 0],
	#		[0, 0, 0, 0, 1, 0, 1, 0],
	#		[0, 1, 1, 0, 0, 0, 1, 0],
	#		[0, 0, 0, 0, 0, 0, 0, 0]]
	grid = [[0, 0, 0, 0, 0],
			[0, 0, 1, 0, 0],
			[1, 1, 1, 0, 0],
			[0, 0, 1, 0, 0],
			[0, 0, 0, 0, 0]]
	numAgents = 1
	numTargets = 1

	model = Model(grid, numAgents, numTargets)
	print "Initial state:"
	model.printWorld()
	timeStep = 0.75
	while True:
		print "\n==============================================================================\n"
		raw_input("Press ENTER for next time step.\n")
		model.update(True)
		if model.isHumanNeeded():
			observedTargets = model.getHumanInput()
			model.refineBelief(observedTargets)