from Model import Model
from World import World
import time
import os

class Stats:
	# Directory name keywords
	POLICY = ["rand", "rtbss", "rtbss-chkpt", "chkpt"]
	RANDOM = 0
	RTBSS = 2
	RTBSS_CHECKPOINT = 3
	CHECKPOINT = 4

	MODEL_REPR = ["sys", "dec-sum"]




	def __init__(self, grid, numAgents, numTargets, policyOption, modelRepr, initState, checkpoints, maxSteps, horizon, discount, timeSteps, humanInput):
		self.model = Model(grid, numAgents, numTargets, policyOption, modelRepr, initState, checkpoints, maxSteps, horizon, discount)

		# Model parameters
		self.policyOption    = policyOption
		self.policy          = Model.POLICIES[policyOption]
		self.discount        = discount
		self.horizon         = horizon
		self.modelReprOption = modelRepr
		self.modelRepr       = Model.MODEL_REPRESENTATIONS[modelRepr]
		self.maxSteps        = maxSteps

		# Simpulation parameters
		self.timeSteps          = timeSteps
		self.simulateHumanInput = humanInput

		# Simulation data
		self.state               = []
		self.agentCompoundState	 = []
		self.targetCompoundState = []
		self.belief              = []
		self.partialBeliefs      = []
		self.reward              = []
		self.calcTime            = []            # Calculation times for each time step
		self.humanEffort         = [0]*timeSteps # Number of times human is asked

	def simulate(self):
		"""Runs simulation."""
		print "Initial state:"
		self.model.printWorld()
		for i in range(self.timeSteps):
			print "Simulating step " + str(i + 1)
			self.state.append(self.model.state)
			agentCompoundState = self.model.state[0]
			targetCompoundState = self.model.state[1]
			self.agentCompoundState.append(agentCompoundState)
			self.targetCompoundState.append(targetCompoundState)
			self.belief.append(self.model.belief)
			self.partialBeliefs.append(self.model.partialBeliefs)
			self.reward.append(self.model.reward)

			# TEST FOR BELIEF AT ACTUAL STATE
			for targetId in range(self.model.numTargets):
				actualTargetState = self.state[i][1][targetId]
				if self.partialBeliefs[i][targetId].get(actualTargetState, None) == None:
					print "actualTargetState: " + str(actualTargetState)
					print "ACTUAL TARGET STATE NOT PART OF PARTIAL BELIEF"
					#raw_input("Press ENTER to continue.")
					#system.exit()
			# Simulate one time step
			#raw_input("Press ENTER for next time step.")
			start = time.time()
			self.model.update()
			if self.simulateHumanInput and self.model.isHumanNeeded():
				observedTargets = self.model.getSimulatedHumanInput()
				self.model.refineBelief(observedTargets)
				self.humanEffort[i] = 1
			elapsedTime = time.time() - start
			self.calcTime.append(elapsedTime)

		print "Simulation of " + str(self.timeSteps) + " time steps finished after " + str(sum(self.calcTime)) + " s."
		print "Policy: " + self.policy
		if self.model.policyOption == Model.RTBSS_CHECKPOINT:
			print "Discount: " + str(self.discount)
			print "Horizon: " + str(self.horizon)
		print "Model representation: " + self.modelRepr
		print "Avg calc time: " + str(sum(self.calcTime)/len(self.calcTime))
		print "Human effort: " + str(sum(self.humanEffort)) + " (" + str(100.0*sum(self.humanEffort)/self.timeSteps) + " %)"
		print "Avg reward: " + str(sum(self.reward)/len(self.reward))

	def getPath(self):
		path = "sim/"

		if self.policyOption == Model.RANDOM:
			path += "rand"
		elif self.policyOption == Model.RTBSS_CHECKPOINT:
			path += "rtbss"
			if self.maxSteps < 10000:
				path += "-cp_maxsteps" + str(self.maxSteps)

			if self.modelReprOption == Model.SYSTEM:
				path += "_sys"
			elif self.modelReprOption == Model.DECENTRALIZED:
				path += "_dec-sum"

			path += "_hor" + str(self.horizon)
		elif self.policyOption == Model.RANDOM_CHECKPOINT:
			path += "chkpt_maxsteps" + str(self.maxSteps)

		if self.model.world.numRobotStates == 34:
			path += "_big"
		elif self.model.world.numRobotStates == 17:
			path += "_small"

		path += "_"
		if not self.simulateHumanInput:
			path += "no"
		path += "human_" + str(self.timeSteps) + "steps"

		path += "_" + str(time.strftime("%y%m%d_%H%M%S", time.localtime())) + "/"

		return path

	def saveData(self, path = None):
		"""Saves data to formatted file for analysis in MATLAB."""
		delimiter = "\n"
		path = self.getPath() if path == None else path
		if not os.path.exists(path):
			os.makedirs(path)
		# Write parameters to file
		params = {"world":				"\n".join([",".join([str(cell) for cell in self.model.world.grid[i]]) for i in range(len(self.model.world.grid))]),
				  "rows":				self.model.world.gridNumRows,
				  "cols":				self.model.world.gridNumCols,
				  "numAgents":			self.model.numAgents,
				  "numTargets":			self.model.numTargets,
				  "policy":				self.policy,
				  "discount":			self.discount,
				  "horizon":			self.horizon,
				  "modelRepr":			self.modelRepr,
				  "timeSteps":			self.timeSteps,
				  "simulateHumanInput":	self.simulateHumanInput,
				  "state":				delimiter.join(str(s) for s in self.state),
				  "reward":				delimiter.join(str(s) for s in self.reward),
				  "calcTime":			delimiter.join(str(s) for s in self.calcTime),
				  "humanEffort":		delimiter.join(str(s) for s in self.humanEffort)}
		for targetId in range(self.model.numTargets):
			partialBeliefContent = ""
			beliefAtStateContent = ""
			beliefAroundStateContent = ""
			for timeStep in range(self.timeSteps):
				partialBelief = self.partialBeliefs[timeStep][targetId]
				for targetState in partialBelief.keys():
					row = targetState[0]
					col = targetState[1]
					p   = partialBelief[targetState]
					partialBeliefContent += str(timeStep) + "," + str(row) + "," + str(col) + "," + str(p) + "\n"
				actualTargetState = self.state[timeStep][1][targetId]
				beliefAtState = partialBelief[actualTargetState] if partialBelief.get(actualTargetState, None) != None else 0.0 # THIS SHOULD NEVER BE ZERO THOUGH!
				beliefAroundState = beliefAtState
				for neighbor in self.model.world.getNeighboringRobotStates(actualTargetState):
					beliefAroundState += partialBelief[neighbor] if partialBelief.get(neighbor, None) != None else 0.0
				beliefAtStateContent += str(beliefAtState) + "\n"
				beliefAroundStateContent += str(beliefAroundState) + "\n"
			
			paramName = "partialBelief-" + str(targetId + 1)
			params[paramName] = partialBeliefContent

			paramName = "beliefAtState-" + str(targetId + 1)
			params[paramName] = beliefAtStateContent

			paramName = "beliefAroundState-" + str(targetId + 1)
			params[paramName] = beliefAroundStateContent


		for paramName in params.keys():
			f = open(path + paramName, "w")
			f.write(str(params[paramName]))
			f.close()

		"""
		f = open(path + "params", "w")
		content = delimiter.join(["rows", "cols", "numAgents", "numTargets", "policy", "discount", "horizon", "modelRepr", "timeSteps", "simulateHumanInput"]) + "\n"
		content += delimiter.join([str(i) for i in [self.model.world.gridNumRows, self.model.world.gridNumCols, self.model.numAgents, self.model.numTargets, self.policy, self.discount, self.horizon, self.modelRepr, self.timeSteps, str(self.simulateHumanInput).lower()]]) + "\n"
		f.write(content)
		f.close()

		# Write results to files
		arraysF = open(path + "arrays", "w")
		#arraysF.write(delimiter.join(["state", "reward", "calcTime", "humanEffort"]) + "\n")
		partialBeliefsF = [None]*self.model.numTargets
		for t in range(self.model.numTargets):
			partialBeliefsF[t] = open(path + "partialBeliefs-" + str(t+1), "w")
			partialBeliefsF[t].write(delimiter.join(["timeStep", "targetState", "probability"]) + "\n")
		for i in range(self.timeSteps):
			arraysF.write(delimiter.join([str(s) for s in [self.state[i], self.reward[i], self.calcTime[i], self.humanEffort[i]]]) + "\n")
			for t in range(self.model.numTargets):
				partialBelief = self.partialBeliefs[i][t]
				for targetState in partialBelief.keys():
					p = str(partialBelief[targetState])
					partialBeliefsF[t].write(delimiter.join([str(i), str(targetState), p]) + '\n')
		for t in range(self.model.numTargets):
			partialBeliefsF[t].close()
		arraysF.close()
		"""

		print "Simulation data successfully saved to path: " + path

if __name__ == '__main__':
	# Set model constants
	GRID = [[0, 0, 0, 0, 0],
			[0, 0, 0, 1, 0],
			[0, 1, 0, 0, 0],
			[0, 0, 0, 1, 0]]

	GRID = [[0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 1, 0, 0, 0],
			[0, 0, 0, 0, 1, 0, 1, 0],
			[0, 1, 1, 0, 0, 0, 1, 0],
			[0, 0, 0, 0, 0, 0, 0, 0]]

	NUM_AGENTS  = 1 # As of now, decentralized RTBSS does not work for multiple agents
	NUM_TARGETS = 2

	POLICY               = Model.RTBSS_CHECKPOINT
	MODEL_REPRESENTATION = Model.DECENTRALIZED
	#MODEL_REPRESENTATION = Model.SYSTEM
	DISCOUNT             = 0.95
	HORIZON              = 2

	INIT_AGENT_STATE  = (World.UPPER_LEFT,)
	#INIT_TARGET_STATE = (World.BOTTOM_RIGHT,)*NUM_TARGETS
	#INIT_TARGET_STATE = ((3,1),(1,4))
	INIT_TARGET_STATE = ((4,2),(2,5))
	INIT_STATE        = (INIT_AGENT_STATE, INIT_TARGET_STATE)

	CHECKPOINTS = [[World.BOTTOM_RIGHT, World.UPPER_LEFT]*NUM_AGENTS]
	MAX_STEPS   = [9999999]#[15]

	# Set simulation constants
	TIME_STEPS      = 5000
	HUMAN_INPUT     = True
	NUM_SIMULATIONS = 3

	for i in range(NUM_SIMULATIONS):
		# Initialize statistics object
		stats = Stats(GRID, NUM_AGENTS, NUM_TARGETS, POLICY, MODEL_REPRESENTATION, INIT_STATE, CHECKPOINTS, MAX_STEPS, HORIZON, DISCOUNT, TIME_STEPS, HUMAN_INPUT)

		print "Running simulation " + str(i) + " for " + stats.getPath()

		# Run simulation
		stats.simulate()

		# Write results to file
		stats.saveData()