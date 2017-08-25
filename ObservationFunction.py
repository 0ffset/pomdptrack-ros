import time
import Util

class ObservationFunction:
	def __init__(self, model):
		self.MAX_REWARD = 100
		self.O = dict()
		self.Ol = dict()
		self.R = dict()
		self.Rl = dict()
		self.targetCompoundStatesFromObservation = dict()
		self.targetStatesFromObservation = dict()
		self.targetStatesFromObservation2 = dict()
		self.model = model
		self.observations = []
		self.observationsAmbiguity = [] # No. of states that correspond to the same observation
		self.directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
		self.orderedCombinationsOfAllTargetsExceptOne = Util.getOrderedCombinations(model.robotStates, model.numTargets - 1) if model.numTargets > 1 else []
		self.observableStates = dict()

		start = time.time()
		print "Initializing observation function..."

		# Each observation is defined as a tuple with on index j either None or the target j's state
		# Each agent can observe in all defined direction until invalid state (wall etc.) is reached
		for state in model.states:
			self.O[state] = dict()
			agentCompoundState, targetCompoundState = state
			self.observableStates[agentCompoundState] = set()
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
						self.observableStates[agentCompoundState].add(observeState)
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
				self.targetStatesFromObservation[observation] = dict()
			else:
				self.observationsAmbiguity[self.observations.index(observation)] += 1
			
			# Map from observation and agent compound state to possible target compound states
			if self.targetCompoundStatesFromObservation[observation].get(agentCompoundState, None) == None:
				self.targetCompoundStatesFromObservation[observation][agentCompoundState] = [targetCompoundState]
			else:
				self.targetCompoundStatesFromObservation[observation][agentCompoundState].append(targetCompoundState)
			
			# Map from observation and agent compound state to possible target states
			if self.targetStatesFromObservation[observation].get(agentCompoundState, None) == None:
				self.targetStatesFromObservation[observation][agentCompoundState] = set()
			for targetState in targetCompoundState:
				self.targetStatesFromObservation[observation][agentCompoundState].add(targetState)
			
			#print "self.targetStatesFromObservation[" + str(observation) + "]" + str(self.targetStatesFromObservation[observation])

			self.O[state][observation] = 1.0
			if numObservedTargets > 0:
				self.R[state] = self.MAX_REWARD/(model.numTargets + 1 - numObservedTargets)

		# Generate partial observation function if system model is decentralized
		if model.modelRepr == model.DECENTRALIZED:
			for sa in model.agentCompoundStates:
				self.Ol[sa] = dict()
				for stl in model.robotStates:
					self.Ol[sa][stl] = dict()
					for o in self.observations:
						# If number of targets is 1, the decentralized representation is equivalent to the system representation
						if self.model.numTargets == 1:
							s = (sa, (stl,))
							self.Ol[sa][stl][o] = self.eval(s, o)
							continue

						# If number of targets is greater than 1
						st = list(self.orderedCombinationsOfAllTargetsExceptOne) # Target compound states
						for i in range(len(st)):
							st[i] = st[i] + (stl,) # Note: value of Ol is independent of l (i.e., probability is same regardless of which individual agent is considered)
						p = 0.0
						for sti in st:
							s = (sa, sti)
							p += self.eval(s, o)
						self.Ol[sa][stl][o] = p/len(self.model.robotStates)
						
						if p > Util.EPSILON:
							if self.targetStatesFromObservation2.get(o, None) == None:
								self.targetStatesFromObservation2[o] = dict()
							if self.targetStatesFromObservation2[o].get(sa, None) == None:
								self.targetStatesFromObservation2[o][sa] = set()
							self.targetStatesFromObservation2[o][sa].add(stl)

		# Perform validy check on observation function
		stop = time.time()
		if self.isValid():
			print "Observation space (" + str(len(self.observations)) + " observations) and observation function initilized successfully in " + str(stop - start) + " s."
		else:
			print "Observation function is invalid!"

	def isValid(self):
		for s in self.model.states:
			observationFcnSuccess = abs(sum(list(self.O[s].values())) - 1.0) < Util.EPSILON
			if not observationFcnSuccess:
				return False
		if self.model.modelRepr == self.model.DECENTRALIZED:
			for sa in self.model.agentCompoundStates:
				for stl in self.model.robotStates:
					partialObservationFcnSuccess = abs(sum(list(self.Ol[sa][stl].values())) - 1.0) < Util.EPSILON
					if not partialObservationFcnSuccess:
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

	def evalOl(self, sa, stl, o):
		"""
		Returns probability of system making observation o given agent
		compound state sa and a single target state stl (l is target index).
		"""
		return self.Ol[sa][stl][o]