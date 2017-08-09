from Model import Model
from World import World
import time

class Stats:
	def __init__(self, grid, numAgents, numTargets, policyOption, modelRepr, initState, checkpoints, maxSteps, horizon, discount, timeSteps, humanInput):
		self.model = Model(grid, numAgents, numTargets, policyOption, modelRepr, initState, checkpoints, maxSteps, horizon, discount)
		self.timeSteps = timeSteps
		self.simulateHumanInput = humanInput

		# Simulation data
		self.state          = []
		self.belief         = []
		self.partialBeliefs = []
		self.reward         = []
		self.calcTimes      = []      # Calculation times for each time step
		self.humanEffort    = 0       # Number of times human is asked

	def simulate(self):
		"""Runs simulation."""
		print "Initial state:"
		self.model.printWorld()
		for i in range(self.timeSteps):
			print "Simulating step " + str(i + 1)
			start = time.time()
			self.state.append(self.model.state)
			self.belief.append(self.model.belief)
			self.partialBeliefs.append(self.partialBeliefs)
			self.reward.append(self.model.reward)
			#raw_input("Press ENTER for next time step.")
			self.model.update(True)
			if self.simulateHumanInput and self.model.isHumanNeeded():
				observedTargets = self.model.getSimulatedHumanInput()
				self.model.refineBelief(observedTargets)
				self.humanEffort += 1
			elapsedTime = time.time() - start
			self.calcTimes.append(elapsedTime)

		print "Simulation of " + str(self.timeSteps) + " time steps finished after " + str(sum(self.calcTimes)) + " s."
		print "Policy: " + Model.POLICIES[self.model.policyOption]
		if self.model.policyOption == Model.RTBSS_CHECKPOINT:
			print "Discount: " + str(self.model.discount)
			print "Horizon: " + str(self.model.horizon)
		print "Model representation: " + Model.MODEL_REPRESENTATIONS[self.model.modelRepr]
		print "Avg calc time: " + str(sum(self.calcTimes)/len(self.calcTimes))
		print "Human effort: " + str(self.humanEffort) + " (" + str(100.0*self.humanEffort/self.timeSteps) + " %)"
		print "Avg reward: " + str(sum(self.reward)/len(self.reward))

	def saveData(self, path):
		"""Saves data to formatted file for analysis in MATLAB."""
		pass

if __name__ == '__main__':
	# Set model constants
	GRID = [[0, 0, 0, 0, 0],
			[0, 0, 1, 0, 0],
			[1, 1, 1, 0, 0],
			[0, 0, 1, 0, 0],
			[0, 0, 0, 0, 0]]

	NUM_AGENTS  = 1
	NUM_TARGETS = 1

	POLICY               = Model.RTBSS_CHECKPOINT
	MODEL_REPRESENTATION = Model.DECENTRALIZED
	DISCOUNT             = 0.95
	HORIZON              = 3

	INIT_AGENT_STATE  = (World.UPPER_LEFT,)
	INIT_TARGET_STATE = (World.RANDOM_STATE,)*NUM_TARGETS
	INIT_STATE        = (INIT_AGENT_STATE, INIT_TARGET_STATE)

	CHECKPOINTS = [[World.BOTTOM_RIGHT, World.UPPER_LEFT]*NUM_AGENTS]
	MAX_STEPS   = [15]

	# Set simulation constants
	TIME_STEPS  = 100
	HUMAN_INPUT = True

	# Initialize statistics object
	stats = Stats(GRID, NUM_AGENTS, NUM_TARGETS, POLICY, MODEL_REPRESENTATION, INIT_STATE, CHECKPOINTS, MAX_STEPS, HORIZON, DISCOUNT, TIME_STEPS, HUMAN_INPUT)

	# Run simulation
	stats.simulate()

	# Write results to file
	path = "sim/simulation_" + str(time.strftime("%y%m%d_%H%M%S", time.localtime()))
	stats.saveData(path)