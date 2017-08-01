import time

class TransitionFunction:
	def __init__(self, model):
		self.T = dict()
		self.model = model

		start = time.time()
		print "Initializing transition funciton..."

		# Generate agent and target transition functions
		agentTransitionFcn = dict()
		targetTransitionFcn = dict()
		for robotState in model.robotStates:

			# Agent transition function
			agentState = robotState
			agentTransitionFcn[agentState] = dict()
			for agentAction in model.agentActions:
				agentTransitionFcn[agentState][agentAction] = dict()
				agentEndState = TransitionFunction.__getRobotEndState(model, agentState, agentAction)
				agentTransitionFcn[agentState][agentAction][agentEndState] = 1.0

			# Target transition function
			targetState = robotState
			targetTransitionFcn[targetState] = dict()
			possibleTargetEndStates = TransitionFunction.__getPossibleTargetEndStates(model, targetState)
			for targetEndState in possibleTargetEndStates:
				targetTransitionFcn[targetState][targetEndState] = 1.0/len(possibleTargetEndStates)

		# Generate agent compound transition function
		agentCompoundTransitionFcn = dict()
		for agentCompoundState in model.agentCompoundStates:
			agentCompoundTransitionFcn[agentCompoundState] = dict()
			for agentCompoundAction in model.actions:
				agentCompoundTransitionFcn[agentCompoundState][agentCompoundAction] = dict()
				agentCompoundEndState = TransitionFunction.__getRobotCompoundEndState(model, agentCompoundState, agentCompoundAction)
				agentCompoundTransitionFcn[agentCompoundState][agentCompoundAction][agentCompoundEndState] = 1.0

		# Generate target compound transition function
		targetCompoundTransitionFcn = dict()
		for targetCompoundState in model.targetCompoundStates:
			targetCompoundTransitionFcn[targetCompoundState] = dict()
			possibleTargetCompoundEndStates = TransitionFunction.__getPossibleTargetCompoundEndStates(model, targetCompoundState)
			for targetCompoundEndState in possibleTargetCompoundEndStates:
				targetCompoundTransitionFcn[targetCompoundState][targetCompoundEndState] = 1.0/len(possibleTargetCompoundEndStates)

		# Generate system transition function
		tot = len(model.states)
		count = 0
		currProc = 0
		for state in model.states:
			proc = int(float(count)/tot*100)
			if proc % 10 == 0 and proc != currProc:
				print str(proc) + " %"
				currProc = proc

			self.T[state] = dict()
			agentCompoundState, targetCompoundState = state[0], state[1]

			for action in model.actions:
				self.T[state][action] = dict()
				agentCompoundEndState = TransitionFunction.__getRobotCompoundEndState(model, agentCompoundState, action)
				possibleTargetCompoundEndStates = TransitionFunction.__getPossibleTargetCompoundEndStates(model, targetCompoundState)

				for targetCompoundEndState in possibleTargetCompoundEndStates:
					pA = agentCompoundTransitionFcn[agentCompoundState][action][agentCompoundEndState]
					pT = targetCompoundTransitionFcn[targetCompoundState][targetCompoundEndState]
					endState = (agentCompoundEndState, targetCompoundEndState)
					self.T[state][action][endState] = pA*pT

			count += 1

		# Perform transition function validity check
		stop = time.time()
		if self.isValid():
			print "Transition function initilized successfully in " + str(stop - start) + " s."
		else:
			print "Transition function is invalid!"

	@staticmethod
	def __getRobotEndState(model, state, action):
		"""Returns end state of robot given initial state and action."""
		endState = (state[0] + action[0], state[1] + action[1])
		return endState if model.world.isValidRobotState(endState) else state

	@staticmethod
	def __getRobotCompoundEndState(model, compoundState, compoundAction):
		"""Returns end compound state of list of agents/targets given list of actions."""
		endCompoundState = tuple()
		for i in range(len(compoundState)):
			state = compoundState[i]
			action = compoundAction[i]
			endState = TransitionFunction.__getRobotEndState(model, state, action)
			endCompoundState += (endState,)
		return endCompoundState

	@staticmethod
	def __getPossibleTargetEndStates(model, targetState):
		"""Returns a set of all possible end states for a given target robot given a initial state."""
		endStates = set()
		for action in model.targetActions:
			endState = TransitionFunction.__getRobotEndState(model, targetState, action)
			endStates.add(endState)
		return endStates

	@staticmethod
	def __getPossibleTargetCompoundEndStates(model, compoundState):
		"""Returns list of possible target compound end states given initial target compound state."""
		compoundEndStates = set()
		for compoundAction in model.targetCompoundActions:
			compoundEndState = TransitionFunction.__getRobotCompoundEndState(model, compoundState, compoundAction)
			compoundEndStates.add(compoundEndState)
		return compoundEndStates

	def isValid(self):
		"""Returns weather transition function is valid or not."""
		for s in self.model.states:
			for a in self.model.actions:
				transitionFcnSuccess = sum(list(self.T[s][a].values())) - 1.0 < 0.000000000001
				if not transitionFcnSuccess:
					return False
		return True

	def getTAsMatrix(self):
		"""Returns transition function as matrix."""
		S = len(self.model.states)
		A = len(self.model.actions)

		start = time.time()
		matrix = [[[0.0 for k in range(S)] for j in range(A)] for i in range(S)]
		for s1 in range(S):
			state = self.model.states[s1]
			for a in range(A):
				action = self.model.actions[a]
				for endState in self.T[state][action].keys():
					s2 = self.model.states.index(endState)
					matrix[s1][a][s2] = self.T[state][action][endState]
		stop = time.time()
		print "Generated transition matrix in " + str(stop - start) + " s."

		return matrix

	def eval(self, s1, a, s2):
		"""Evaluates transition function for given initial state, action and end state."""
		if self.T.get(s1, None) != None:
			if self.T[s1].get(a, None) != None:
				if self.T[s1][a].get(s2, None) != None:
					return self.T[s1][a][s2]
		#if s1 not in self.model.states:
		#	print "Initial state invalid during transition function evaluation: " + str(s1)
		#	return None
		#if s2 not in self.model.states:
		#	print "End state invalid during transition function evaluation: " + str(s2)
		#	return None
		#if a not in self.model.actions:
		#	print "Action invalid during transition function evaluation: " + str(a)
		#	return None
		#print "Transition function evaluated as 0. Using optimal implementation, this evaluation should be unnecessary."
		return 0.0