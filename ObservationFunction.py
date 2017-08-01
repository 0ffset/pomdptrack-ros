import time

class ObservationFunction:
	def __init__(self, model):
		self.MAX_REWARD = 100
		self.O = dict()
		self.R = dict()
		self.targetCompoundStatesFromObservation = dict()
		self.model = model
		self.observations = []
		self.observationsAmbiguity = [] # No. of states that correspond to the same observation
		self.directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

		start = time.time()
		print "Initializing observation function..."

		# Each observation is defined as a tuple with on index j either None or the target j's state
		# Each agent can observe in all defined direction until invalid state (wall etc.) is reached
		for state in model.states:
			self.O[state] = dict()
			agentCompoundState, targetCompoundState = state[0], state[1]
			observation = [None for i in range(model.numTargets)]
			for agentState in agentCompoundState:
				
				observeState = agentState
				for i in range(len(targetCompoundState)):
					targetState = targetCompoundState[i]
					if observeState == targetState:
						observation[i] = targetState
				
				for direction in self.directions:
					observeState = (agentState[0] + direction[0], agentState[1] + direction[1])
					while model.world.isValidRobotState(observeState):
						for i in range(len(targetCompoundState)):
							targetState = targetCompoundState[i]
							if observeState == targetState:
								observation[i] = targetState
						observeState = (observeState[0] + direction[0], observeState[1] + direction[1])
			
			numObservedTargets = model.numTargets - observation.count(None)
			observation = frozenset(set(observation) - set((None,)))
			if observation not in self.observations:
				self.observations.append(observation)
				self.observationsAmbiguity.append(1)
				self.targetCompoundStatesFromObservation[observation] = dict()
			else:
				self.observationsAmbiguity[self.observations.index(observation)] += 1
			
			if agentCompoundState not in self.targetCompoundStatesFromObservation[observation].keys():
				self.targetCompoundStatesFromObservation[observation][agentCompoundState] = [targetCompoundState]
			else:
				self.targetCompoundStatesFromObservation[observation][agentCompoundState].append(targetCompoundState)

			self.O[state][observation] = 1.0
			#if self.R.get(state, None) == None:
			#	self.R[state] = dict()
			if numObservedTargets > 0:
				self.R[state] = self.MAX_REWARD/(model.numTargets + 1 - numObservedTargets)

		# Perform validy check on observation function
		stop = time.time()
		if self.isValid():
			print "Observation space (" + str(len(self.observations)) + " observations) and observation function initilized successfully in " + str(stop - start) + " s."
		else:
			print "Observation function is invalid!"

	def isValid(self):
		for s in self.model.states:
			observationFcnSuccess = sum(list(self.O[s].values())) - 1.0 < 0.00000000001
			if not observationFcnSuccess:
				return False
		return True

	def printObservationsAndAbiguityCount(self):
		for i in range(len(self.observations)):
			print str(self.observations[i]) + ": " + str(self.observationsAbiguity[i])

	def getObservations(self):
		"""Returns set of all observations."""
		return self.observations

	def getRewardFunction(self):
		"""Returns the reward function."""
		return self.R

	def getOAsMatrix(self):
		"""Returns observation function as matrix."""
		S = len(self.model.states)
		A = len(self.model.actions)
		O = len(self.model.observations)

		start = time.time()
		matrix = [[[0.0 for k in range(O)] for j in range(A)] for i in range(S)]
		for s in range(S):
			state = self.model.states[s]
			for a in range(A):
				for observation in self.O[state].keys():
					o = self.observations.index(observation)
					matrix[s][a][o] = self.O[state][observation]
		stop = time.time()
		print "Generated observation matrix in " + str(stop - start) + " s."

		return matrix

	def eval(self, s, o):
		"""Evaluates transition function for given initial state, action and end state."""
		if self.O.get(s, None) != None:
			if self.O[s].get(o, None) != None:
				return self.O[s][o]
		#if s not in self.model.states:
		#	print "State invalid during transition function evaluation."
		#	return None
		#if o not in self.model.observations:
		#	print "End state invalid during transition function evaluation."
		#	return None
		#print "Observation function evaluated as 0. Using optimal implementation, this evaluation should be unnecessary."
		return 0.0