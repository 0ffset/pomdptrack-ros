import time
import numpy.random
import Util
from multiprocessing.pool import ThreadPool

class Solver:
	def __init__(self, model, discount = 0.95, horizon = 4):
		self.model = model
		self.discount = discount
		self.horizon = horizon
		self.minSteps = dict()
		if model.modelRepr == model.SYSTEM:
			self.observations = model.observations
		elif model.modelRepr == model.DECENTRALIZED:
			self.observations = model.agentObservations
		self.__initMinSteps()
		self.numRecursions = 0
		self.pool = ThreadPool()

	def __initMinSteps(self):
		"""Initializes the minimum steps function."""
		timeStart = time.time()
		for start in self.model.robotStates:
			for end in self.model.robotStates:
				self.minSteps[(start, end)] = self.getMinSteps(start, end)
		timeElapsed = time.time() - timeStart
		print "Calculated shortest paths in " + str(timeElapsed) + " s."

	def getMinSteps(self, start, end):
		"""Retrurns minimum steps required to get from start to end state in the grid."""
		if (start, end) in self.minSteps.keys():
			return self.minSteps[(start, end)]
		elif (end, start) in self.minSteps.keys():
			return self.minSteps[(end, start)] # shortest way there is also shortest way back reversed
		else:
			return self.__BFSMinSteps(start, end)

	def __BFSMinSteps(self, start, end):
		"""Returns minimum steps required to get from start to end using BFS."""
		visited = set()
		queue = [(start, 0)]
		while len(queue) > 0:
			node = queue.pop(0)
			state = node[0]
			numSteps = node[1]
			if state == end:
				return numSteps
			if state not in visited:
				visited.add(state)
				self.minSteps[(start, state)] = numSteps
				self.minSteps[(state, start)] = numSteps
				for action in self.model.agentActions:
					nextState = self.__getRobotEndState(state, action)
					queue.append((nextState, numSteps + 1))

	def __getRobotEndState(self, state, action):
		"""Return robot end state after taking action in initial state."""
		endState = (state[0] + action[0], state[1] + action[1])
		return endState if self.model.world.isValidRobotState(endState) else state

	def getAllowedActions(self, agentCompoundState = None, checkpoints = None, checkpointIndeces = None, maxSteps = None):
		"""Returns actions that allow all agents to reach their checkpoints in time."""
		agentCompoundState = self.model.state[0] if agentCompoundState == None else agentCompoundState
		checkpoints = self.model.checkpoints if checkpoints == None else checkpoints
		checkpointIndeces = self.model.checkpointIndeces if checkpointIndeces == None else checkpointIndeces
		maxSteps = self.model.maxSteps if maxSteps == None else maxSteps
		
		# Determine allowed individual actions for each agent 
		allowedAgentActions = [None for i in range(self.model.numAgents)]
		for i in range(self.model.numAgents):
			allowedAgentActions[i] = []
			agentState = agentCompoundState[i]
			checkpointIndex = checkpointIndeces[i]
			end = checkpoints[i][checkpointIndex]
			for agentAction in self.model.agentActions:
				start = self.__getRobotEndState(agentState, agentAction)
				maximum = maxSteps[i]
				if self.getMinSteps(start, end) <= maximum:
					allowedAgentActions[i].append(agentAction)

		# Construct all allowed system actions from individual agent actions
		allowedActions = Util.getCombinationsDifferentSets(allowedAgentActions)

		return allowedActions

	def getRandomAllowedAction(self, agentCompoundState = None, checkpoints = None, checkpointIndeces = None, maxSteps = None):
		"""Returns random action that allows all agents to reach their checkpoints in time."""
		agentCompoundState = self.model.state[0] if agentCompoundState == None else agentCompoundState
		checkpoints = self.model.checkpoints if checkpoints == None else checkpoints
		checkpointIndeces = self.model.checkpointIndeces if checkpointIndeces == None else checkpointIndeces
		maxSteps = self.model.maxSteps if maxSteps == None else maxSteps
		allowedActions = self.getAllowedActions(agentCompoundState, checkpoints, checkpointIndeces, maxSteps) # Determine actions allowed to take in order to get to next checkpoint in time
		action = self.getRandomAction(allowedActions)
		return action

	def getRandomAction(self, actions = None):
		"""Returns a random action."""
		actions = self.model.actions if actions == None else actions
		a = numpy.random.choice(len(actions))
		action = actions[a]
		return action

	def getActionsWithUniqueEndStates(self, actions, agentCompoundState):
		"""Returns list of actions that yield unique end agent compound states."""
		uniqueActions = []
		agentCompoundEndStates = set()

		for action in actions:
			agentCompoundEndState = self.model.Ta(agentCompoundState, action)
			if agentCompoundEndState not in agentCompoundEndStates:
				uniqueActions.append(action)
				agentCompoundEndStates.add(agentCompoundEndState)

		return uniqueActions

	def checkpointDecentralizedRTBSS(self):
		"""Returns optimal action based on decentralized RTBSS with checkpoints."""
		start = time.time()
		actionValuesList = []
		actionSumValues = dict()
		for action in self.model.actions:
			actionSumValues[action] = 0.0
		optimalAction = None
		optimalValue = -float('inf')
		i = 0
		
		#actionValuePairs = self.pool.map(self.checkpointRTBSS, self.model.partialBeliefs)
		#optimalAction, optimalValue = actionValuePairs[0]
		#for a, v in actionValuePairs:
		#	if v > optimalValue:
		#		optimalValue = v
		#		optimalAction = a
		
		for partialBelief in self.model.partialBeliefs:
			if self.model.doPrint:
				print "Performing RTBSS for target " + str(i) + " belief..."
			action, value = self.checkpointRTBSS(partialBelief)
			actionValuesList.append(self.actionValues)
			for a in self.actionValues.keys():
				actionSumValues[a] += self.actionValues[a]
			if value > optimalValue:
				optimalValue = value
				optimalAction = action
			i += 1
		#print "actionSumValues: " + str(actionSumValues)

		# TEST: USE SUM OF BOTH VALUES TO DETERMINE ACTION
		optimalAction = max(actionSumValues.iterkeys(), key = (lambda key: actionSumValues[key]))
		optimalValue = max(actionSumValues.values())
		
		elapsedTime = time.time() - start
		if self.model.doPrint:
			print "Optimal action based on decentralized RTBSS with horizon " + str(self.horizon) + " found in " + str(elapsedTime) + " s."
			print "Action: " + str(optimalAction) + str(", value: ") + str(optimalValue)
		return optimalAction, optimalValue

	def checkpointRTBSS(self, belief = None):
		"""Returns optimal system action based on RTBSS with checkpoints."""
		belief = self.model.belief if belief == None else belief
		self.numRecursions = 0

		start = time.time()
		self.actionValues = dict()
		for action in self.model.actions:
			self.actionValues[action] = -float('inf')

		# Perform RTBSS
		self.__checkpointRTBSS(self.model.state[0], dict(belief), int(self.horizon), self.model.checkpoints, list(self.model.checkpointIndeces), list(self.model.maxSteps), 0.0)
		if self.model.doPrint:
			print "Number of recursions: " + str(self.numRecursions)

		# Pick out optimal action (action with highest value) and return
		optimalAction = max(self.actionValues.iterkeys(), key = (lambda key: self.actionValues[key]))
		optimalValue = max(self.actionValues.values())
		elapsedTime = time.time() - start
		if self.model.doPrint:
			print self.actionValues
			print "Optimal action based on RTBSS with horizon " + str(self.horizon) + " found in " + str(elapsedTime) + " s."
		return optimalAction, optimalValue

	def __checkpointRTBSS(self, agentCompoundState, belief, depth, checkpoints, checkpointIndeces, maxSteps, accExpReward):
		"""Helper function for recursive belief tree search in RTBSS-based solver."""
		self.numRecursions += 1

		if depth == 0:
			finalValue = accExpReward + self.discount**self.horizon*self.getExpectedReward(agentCompoundState, belief)
			return finalValue

		accExpReward += self.discount**(self.horizon - depth)*self.getExpectedReward(agentCompoundState, belief)

		# Only consider actions that allow agents to get to their checkpoints in time
		allowedActions = self.getAllowedActions(agentCompoundState, checkpoints, checkpointIndeces, maxSteps)

		# Only consider actions that yields unique end states
		allowedActions = self.getActionsWithUniqueEndStates(allowedActions, agentCompoundState)

		# Update checkpoints
		checkpointIndeces, maxSteps = self.model.updateCheckpoints(checkpoints, checkpointIndeces, maxSteps)
		maxSteps = [maxSteps[i] - 1 for i in range(len(maxSteps))]

		for action in allowedActions:
			if depth == self.horizon:
				if self.model.doPrint:
					print "RTBSS for action " + str(action)
			expReward = 0.0
			for observation in self.observations: #"""CAN WE LOOP OVER SUBSET?"""
				agentCompoundEndState = self.model.Ta(agentCompoundState, action)
				#start = time.time()
				newUnnormalizedBelief = self.getNewUnnormalizedBelief(belief, action, observation, agentCompoundState, agentCompoundEndState)
				#print "Time calc belief: " + str(time.time() - start)

				if newUnnormalizedBelief == False: # This means that observation is impossible given belief and action
					continue
				observationProbability = sum(newUnnormalizedBelief.values())
				newBelief = self.model.getNormalizedNonZeroBelief(newUnnormalizedBelief)

				expReward += self.discount**(self.horizon - depth)*observationProbability*self.__checkpointRTBSS(agentCompoundEndState, newBelief, depth - 1, checkpoints, checkpointIndeces, maxSteps, accExpReward)
				
			if depth == self.horizon:
				if self.model.doPrint:
					print "Horizon reached for action " + str(action)
				if expReward > self.actionValues[action]:
					self.actionValues[action] = expReward

		return expReward

	def getNewUnnormalizedBelief(self, belief, action, observation, agentCompoundState, agentCompoundEndState):
		"""Returns an updated unnormalized belief."""
		if self.model.modelRepr == self.model.SYSTEM:
			return self.model.getNewUnnormalizedBelief(belief, action, observation, agentCompoundState, agentCompoundEndState)
		elif self.model.modelRepr == self.model.DECENTRALIZED:
			return self.model.getNewUnnormalizedPartialBelief(belief, observation, agentCompoundEndState)

	def getExpectedReward(self, agentCompoundState = None, belief = None):
		"""Returns expected reward based on belief."""
		agentCompoundState = self.model.state[0] if agentCompoundState == None else agentCompoundState
		belief = self.model.belief if belief == None else belief
		expReward = 0.0
		for targetCompoundState in belief.keys():
			state = (agentCompoundState, targetCompoundState)
			if self.model.modelRepr == self.model.SYSTEM:
				expReward += belief[targetCompoundState]*self.model.R(state)
			elif self.model.modelRepr == self.model.DECENTRALIZED:
				expReward += belief[targetCompoundState]*self.model.Rl(state)
		return expReward

	def getBeliefObservationProbability(self, agentCompoundState, belief, action, observation):
		"""
		Returns probability of next transition making an observation given current belief.
		SINCE THIS PROBABILITY IS THE SAME AS THE SUM OF THE UNNORMALIZED UPDATED BELIEF,
		THIS METHOD IS UNUSED IN THE RTBSS ALGORITHM.
		"""
		start = time.time()
		# Determine possible initial states
		possibleStates = set()
		for targetCompoundState in belief.keys():
			state = (agentCompoundState, targetCompoundState)
			possibleStates.add(state)

		# Determine possible end states
		agentCompoundEndState = self.model.Ta(agentCompoundState, action)
		possibleEndStates = set()
		for targetCompoundState in belief.keys():
			for targetCompoundEndState in self.model.Tt(targetCompoundState).keys():
				endState = (agentCompoundEndState, targetCompoundEndState)
				possibleEndStates.add(endState)
		
		# Calculate observation probability
		observationProbability = 0.0
		for endState in possibleEndStates:
			endStateProbability = 0.0
			for state in possibleStates:
				targetState = state[1]
				endStateProbability += self.model.T(state, action, endState)*belief[targetState]
			observationProbability += self.model.O(endState, observation)*endStateProbability

		elapsedTime = time.time() - start
		print "Belief observation probability calculated in " + str(elapsedTime) + " s."
		print "Observation probability: " + str(observationProbability)

		return observationProbability