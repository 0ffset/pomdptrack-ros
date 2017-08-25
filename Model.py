from World import World
from Solver import Solver
from TransitionFunction import TransitionFunction
from ObservationFunction import ObservationFunction
import Util
import time
import numpy.random

class Model:
	# Policy options
	POLICIES = ["Random", "Random checkpoint", "RTBSS checkpoint"]
	RANDOM            = 0
	RANDOM_CHECKPOINT = 1
	RTBSS_CHECKPOINT  = 2

	# Model representation
	MODEL_REPRESENTATIONS = ["System", "Decentralized"]
	SYSTEM        = 0
	DECENTRALIZED = 1
	
	def __init__(self, grid = None, numAgents = None, numTargets = None, policyOption = None, modelRepr = None, initState = None, checkpoints = None, maxSteps = None, horizon = None, discount = None):
		self.world             = World(grid)
		self.numAgents         = 1                              if numAgents == None    else numAgents
		self.numTargets        = 1                              if numTargets == None   else numTargets
		self.policyOption      = Model.RTBSS_CHECKPOINT         if policyOption == None else policyOption
		self.modelRepr         = Model.SYSTEM                   if modelRepr == None    else modelRepr
		self.state             = self.__getDefaultInitState()   if initState == None    else self.getInitState(initState)
		self.checkpoints       = self.__getDefaultCheckpoints() if checkpoints == None  else self.getCheckpoints(checkpoints)
		self.checkpointIndeces = [0 for i in range(self.numAgents)]
		self.DEFAULT_MAX_STEPS = [20 for i in range(self.numAgents)]
		self.maxSteps          = self.__getDefaultMaxSteps()    if maxSteps == None     else maxSteps
		self.horizon           = 1                              if horizon == None      else horizon
		self.discount          = 0.95                           if discount == None      else discount
		self.reward            = 0

		# Initialize sets of all actions
		self.agentActionLabels     = ["stay", "go N", "go E", "go S", "go W"]
		self.agentActions          = [(0, 0), (-1, 0), (0, 1), (1, 0), (0, -1)]
		self.targetActionLabels    = ["stay", "go N", "go E", "go S", "go W"]
		self.targetActions         = [(0, 0), (-1, 0), (0, 1), (1, 0), (0, -1)]
		#self.targetActions         = [(0, 0), (-1, 0), (0, 1), (1, 0), (0, -1), (-2, 0), (0, 2), (2, 0), (0, -2)]
		#self.targetActions         = [(0, 0), (-1, 0), (0, 1), (1, 0), (0, -1), (-2, 0), (0, 2), (2, 0), (0, -2), (-3, 0), (0, 3), (3, 0), (0, -3)]
		#self.targetActions         = [(0, 0)]
		self.actions               = Util.cartesianPower(self.agentActions, self.numAgents) #Util.getOrderedCombinations(self.agentActions, self.numAgents)
		self.targetCompoundActions = Util.cartesianPower(self.targetActions, self.numTargets) #Util.getOrderedCombinations(self.targetActions, self.numTargets)
		
		# Initialize lists of all states
		self.robotStates          = self.world.robotStates
		self.agentCompoundStates  = []
		self.targetCompoundStates = []
		self.states               = []
		self.__initStates()

		# Initialize transition, observation and reward functions
		self.transitionFcn        = TransitionFunction(self)
		self.observationFcn       = ObservationFunction(self)
		self.agentObservations    = [frozenset([robotState]) for robotState in self.robotStates]
		self.observations         = self.observationFcn.getObservations()
		self.rewardFcn            = self.observationFcn.getRewardFunction() # Reward function is generated upon observation function init
		self.observation          = frozenset() # Last set of observations
		self.observationList      = [] # Last observation as enumerated set
		self.ambiguousObservation = [] # Last subset of observation that is ambiguous
		if self.modelRepr == Model.DECENTRALIZED:
			self.partialRewardFcn = dict()
			self.__initPartialRewardFcn()

		# Initialize belief
		self.belief = dict()
		self.__initBelief()

		# Initialize policy solver
		self.solver = Solver(self, self.discount, self.horizon)

		# If results should be printed in terminal
		self.doPrint = False

	def getInitState(self, initStateInput):
		"""Get the initial state of the system from input on initialization."""
		numRobots = [self.numAgents, self.numTargets]
		initState = tuple()
		print "getInitState:"
		for i in range(2):
			initCompoundStateInput = initStateInput[i]
			initCompoundState = tuple()
			for j in range(len(initCompoundStateInput)):
				initRobotStateInput = initCompoundStateInput[j]
				if type(initRobotStateInput) is int:
					keyword = initRobotStateInput
					initCompoundState += (self.world.getState(keyword),)
				else:
					initCompoundState += (initRobotStateInput,)
			initState += (initCompoundState,)
		print initState
		return initState

	def getCheckpoints(self, checkpointsInput):
		"""Returns correctly formatted checkpoints list from input."""
		for i in range(len(checkpointsInput)):
			for j in range(len(checkpointsInput[i])):
				if type(checkpointsInput[i][j]) is int:
					keyword = checkpointsInput[i][j]
					checkpointsInput[i][j] = self.world.getState(keyword)
		return checkpointsInput


	def __getDefaultInitState(self):
		"""Generates and returns a default initial system state."""
		agentCompoundState = tuple((i, 0) for i in range(self.numAgents))
		targetCompoundState = tuple((i, self.world.gridNumCols-1) for i in range(self.numTargets))
		return (agentCompoundState, targetCompoundState)

	def __getDefaultCheckpoints(self):
		"""Generates and returns a default list of checkpoints"""
		checkpoints = [[] for i in range(self.numAgents)]
		for i in range(self.numAgents):
			state1 = (self.world.gridNumRows - 1, self.world.gridNumCols - 1)
			state2 = (0, 0)
			checkpoints[i] = [state1, state2]
		return checkpoints

	def __getDefaultMaxSteps(self):
		"""Returns default max steps between checkpoints."""
		maxSteps = [None for i in range(self.numAgents)]
		for i in range(self.numAgents):
			maxSteps[i] = self.DEFAULT_MAX_STEPS[i]
		return maxSteps

	def __initStates(self):
		"""Initializes all states system and sub-system states."""
		start = time.time()
		print "Initializing state space..."

		# Enumerate agent and target compound states
		self.agentCompoundStates = Util.cartesianPower(self.robotStates, self.numAgents) #Util.getOrderedCombinations(self.robotStates, self.numAgents)
		self.targetCompoundStates = Util.cartesianPower(self.robotStates, self.numTargets) #Util.getOrderedCombinations(self.robotStates, self.numTargets)

		# Enumerate system states
		#for agentCompoundState in self.agentCompoundStates:
		#	for targetCompoundState in self.targetCompoundStates:
		#		self.states.append((agentCompoundState, targetCompoundState))
		self.states = Util.cartesianProduct(self.agentCompoundStates, self.targetCompoundStates)

		stop = time.time()
		print "State space (" + str(len(self.states)) + " states) initialized successfully in " + str(stop - start) + " s."

	def __getRobotEndState(self, state, action):
		"""Return robot end state after taking action in initial state."""
		endState = (state[0] + action[0], state[1] + action[1])
		return endState if self.world.isValidRobotState(endState) else state


	def __initBelief(self):
		"""Initialize belief state."""
		for targetCompoundState in self.targetCompoundStates:
			self.belief[targetCompoundState] = 1.0/len(self.targetCompoundStates)
		self.partialBeliefs = self.getPartialBeliefs()

	def T(self, s1, a, s2 = None):
		"""
		Evaluates transition function for given initial state, action and end state,
		i.e. returns probability of system ending up in s2 after starting in s1 and taking action a
		"""
		#if s2 == None:
		#	PT = self.transitionFcn.T[s1][a]
		#	if len(PT) == 1:
		#		return PT.keys()[0]
		#	return PT
		return self.transitionFcn.eval(s1, a, s2)

	def Ta(self, s1, a, s2 = None):
		"""
		Evaluates agent compound transition function for given initial state, action and end state,
		i.e. returns probability of agents ending up in s2 after starting in s1 and taking action a
		"""
		if s2 == None:
			PT = self.transitionFcn.Ta[s1][a]
			if len(PT) == 1:
				return PT.keys()[0]
			return PT
		return self.transitionFcn.evalTa(s1, a, s2)

	def Tt(self, s1, s2 = None):
		"""
		Evaluates agent compound transition function for given initial state, action and end state,
		i.e. returns probability of agents ending up in s2 after starting in s1 and taking action a
		"""
		if s2 == None:
			PT = self.transitionFcn.Tt[s1]
			if len(PT) == 1:
				return PT.keys()[0]
			return PT
		return self.transitionFcn.evalTt(s1, s2)

	def Ttl(self, stl1, stl2):
		"""
		Evaluates the partial target transition function for given initial and end target states.
		i.e. returns probability of one target ending up in stl2 starting in stl1 
		"""
		#if stl2 == None:
		#	PT = self.transitionFcn.Ttl[stl1]
		#	if len(PT) == 1:
		#		return PT.keys()[0]
		#	return PT
		return self.transitionFcn.evalTtl(stl1, stl2)

	def O(self, s, o):
		"""
		Evaluates observation function for given state and observation,
		i.e. returns probability of system making observation o in state s
		"""
		if o == None:
			PO = self.observationFcn.O[s]
			if len(PO) == 1:
				return PO.keys()[0]
			return PO
		return self.observationFcn.eval(s, o)

	def Ol(self, sa, stl, o):
		"""
		Evaliates partial observation for given agent compound state, individual target state,
		target index and observation, i.e. returns probability of system making observation o
		given agent compound state sa etc.
		"""
		return self.observationFcn.evalOl(sa, stl, o)

	def R(self, s):
		"""Returns the reward associated with a system state."""
		penalty = -0.5
		if self.rewardFcn.get(s, None) != None:
			return self.rewardFcn[s]
		return penalty

	def __initPartialRewardFcn(self):
		"""Initializes partial reward function."""
		print "Initializing partial reward function..."
		start = time.time()
		for stl in self.robotStates:
			st = list(self.observationFcn.orderedCombinationsOfAllTargetsExceptOne) # Target compound states
			for i in range(len(st)):
				st[i] = st[i] + (stl,)
			for sa in self.agentCompoundStates:
				sl = (sa, stl)
				if self.numTargets == 1:
					s = (sa, (stl,))
					self.partialRewardFcn[sl] = self.R(s)
					continue
				reward = 0.0
				for sti in st:
					s = (sa, sti)
					reward += self.R(s)
				self.partialRewardFcn[sl] = reward/len(st)
		elapsedTime = time.time() - start
		print "Partial reward function initialized in " + str(elapsedTime) + " s."

	def Rl(self, sl):
		"""Returns the partial reward associated with one of the targets being in state stl. sl = (sa, stl)."""
		return self.partialRewardFcn[sl]

	def updateCheckpoints(self, checkpoints = None, checkpointIndeces = None, maxSteps = None):
		"""Updates checkpoint to next one for each agent if max allowed steps left is zero."""
		checkpoints = self.checkpoints if checkpoints == None else checkpoints
		checkpointIndeces = list(self.checkpointIndeces) if checkpointIndeces == None else list(checkpointIndeces)
		maxSteps = list(self.maxSteps) if maxSteps == None else list(maxSteps)
		for i in range(self.numAgents):
			checkpointIndex = checkpointIndeces[i]
			if maxSteps[i] == 0:
				maxSteps[i] = self.DEFAULT_MAX_STEPS[i]
				checkpointIndeces[i] = (checkpointIndex + 1)%len(checkpoints[i])
		return checkpointIndeces, maxSteps

	def __sampleState(self, state, action):
		"""Samples a system end state given initial state and action."""
		agentCompoundState, targetCompoundState = state[0], state[1]
		agentCompoundEndState = self.Ta(agentCompoundState, action)
		PTt = self.transitionFcn.Tt[targetCompoundState]
		st = numpy.random.choice(range(len(PTt)), p = PTt.values())
		targetCompoundEndState = PTt.keys()[st]
		endState = (agentCompoundEndState, targetCompoundEndState)
		return endState

	def __sampleObservation(self, state):
		"""Samples an observation given the state."""
		PO = self.observationFcn.O[self.state]
		o = numpy.random.choice(range(len(PO)), p = PO.values())
		observation = PO.keys()[o]
		return observation

	def update(self, doPrint = False):
		"""
		Calculates optimal action for current belief, updates system state, makes observation
		and updates belief state.
		"""
		self.doPrint = doPrint
		self.printList = [] # List for simultaneous print later (to avoid twitchy print)
		if doPrint:
			print "\nPerforming system update..."

		# Calculate optimal action
		self.maxSteps = [self.maxSteps[i] - 1 for i in range(len(self.maxSteps))] # Decrement max allowed steps by 1 for all agents
		if self.policyOption == Model.RANDOM:
			action = self.solver.getRandomAction()
		elif self.policyOption == Model.RANDOM_CHECKPOINT:
			action = self.solver.getRandomAllowedAction()
		elif self.policyOption == Model.RTBSS_CHECKPOINT:
			if self.modelRepr == Model.SYSTEM:
				if doPrint:
					print "Performing RTBSS for system representation..."
				action, value = self.solver.checkpointRTBSS()
			elif self.modelRepr == Model.DECENTRALIZED:
				if doPrint:
					print "Performing RTBSS for decentralized representation..."
				action, value = self.solver.checkpointDecentralizedRTBSS()
		
		# Update checkpoints
		self.checkpointIndeces, self.maxSteps = self.updateCheckpoints()
		nextCheckpoints = [None for i in range(self.numAgents)]
		agentActionLabel = [None for i in range(self.numAgents)]
		for i in range(self.numAgents):
			nextCheckpoints[i] = self.checkpoints[i][self.checkpointIndeces[i]]
			agentActionLabel[i] = self.agentActionLabels[self.agentActions.index(action[i])]
		self.printList.append("Action: " + str(agentActionLabel))
		self.printList.append("Next checkpoint: " + str(nextCheckpoints))
		self.printList.append("Max steps left: " + str(self.maxSteps))

		# Update system state
		agentCompoundState, targetCompoundState = self.state[0], self.state[1]
		self.state = self.__sampleState(self.state, action)
		self.printList.append("State: " + str(self.state))
		self.reward = self.R(self.state)
		self.printList.append("Reward: " + str(self.reward))

		# Make observation
		self.observation = self.__sampleObservation(self.state)
		self.observationList = list(self.observation) # Enumarate all observations
		self.printList.append("Made observation: " + str(self.observation))

		# Update belief
		agentCompoundEndState = self.state[0]
		start = time.time()
		if self.modelRepr == Model.SYSTEM:
			self.belief = self.getNewBelief(self.belief, action, self.observation, agentCompoundState, agentCompoundEndState, self.printList)
			self.partialBeliefs = self.getPartialBeliefs()
		elif self.modelRepr == Model.DECENTRALIZED:
			self.partialBeliefs = [self.getNewPartialBelief(self.partialBeliefs[i], self.observation, agentCompoundEndState, self.printList) for i in range(len(self.partialBeliefs))]
		stop = time.time()
		#self.printList.append("Exp. reward: " + str(self.solver.getExpectedReward()))
		self.printList.append("Belief updated in " + str(stop - start) + " s.")

		# Update partial belief
		self.printList.append("Max belief: " + str(max(self.belief.values())))
		self.printList.append("Number of states in belief: " + str(len(self.belief)))
		self.printList.append("Individual beliefs:")

		# Refine belief based on deduction that some observations can only correspond to certain targets
		self.ambiguousObservation = [] # Subset of the observation that agent can't identify with specific targets
		if len(self.observation) > 0:
			observedTargets = self.deduceWhichTargets()
			#observedTargets = [None for i in range(len(self.observationList))] # Targets observed in observation i
			#for i in range(len(self.observationList)):
			#	observedState = self.observationList[i]
			#	targetIndex = self.deduceWhichTarget(observedState)
			#	canDeduce = targetIndex != None
			#	if canDeduce:
			#		observedTargets[i] = [targetIndex]
			#	else:
			#		self.ambiguousObservation.append(observedState)
			# Refine belief if targets locations are deduced from observation
			if observedTargets.count(None) != len(observedTargets):
				self.refineBelief(observedTargets)
			for i in range(3):
				self.printList.pop()
			self.printList.append("Max belief: " + str(max(self.belief.values())))
			self.printList.append("Number of states in belief: " + str(len(self.belief)))
			self.printList.append("Individual beliefs:")

		for i in range(self.numTargets):
			self.printList.append("\tTarget " + str(i+1) + ": " + str(len(self.partialBeliefs[i])))

		#"""TEST BELIEF OBSERVATION PROBABILITY"""
		#obsProb = str(self.solver.getBeliefObservationProbability(agentCompoundState, self.belief, action, self.observation))
		#self.printList.append("Probability of making same observation next step: " + obsProb)

		# Print results
		if doPrint:
			self.printUpdateResults()

	def deduceWhichTargets(self):
		"""
		Deduces which targets correspond to which observations by considering possible states in belief.
		If a state has non-zero probability only for one target, that target has to be in that state.
		"""
		observedStates = set(self.observationList)
		possibleTargetStates = [set(self.partialBeliefs[targetId].keys()) for targetId in range(self.numTargets)]
		possibleTargets = set(i for i in range(self.numTargets))
		observedTargets = [None for i in range(len(self.observationList))]
		canDeduce = True
		#for i in range(len(self.partialBeliefs)):
		#	print "partialBelief " + str(i+1) + ":"
		#	print "\t" + str(self.partialBeliefs[i])
		while canDeduce:
			canDeduce = False
			deductions = set()
			for observedState in observedStates:
				targetsPossiblyAtObservedState = []
				for i in possibleTargets:
					#print "target: " + str(i+1)
					#print "possibleTargetStates[i]: " + str(possibleTargetStates[i])
					#print "observedState: " + str(observedState)
					if observedState in possibleTargetStates[i]:
						targetsPossiblyAtObservedState.append(i)
				if len(targetsPossiblyAtObservedState) == 1:
					targetId = targetsPossiblyAtObservedState[0]
					if self.doPrint:
						print "Only target " + str(targetId+1) + " can be in " + str(observedState)
					canDeduce = True
					observedTargets[self.observationList.index(observedState)] = [targetId]
					#print "possibleTargets: " + str(possibleTargets)
					possibleTargets.remove(targetId)
					#print "possibleTargets: " + str(possibleTargets) + " (after removal)"
					#observedStates.remove(observedState)
					deductions.add(observedState)
			for observedState in deductions:
				observedStates.remove(observedState)
		for i in range(len(observedTargets)):
			if observedTargets[i] == None:
				self.ambiguousObservation.append(self.observationList[i])
		return observedTargets

	def deduceWhichTarget(self, observedState):
		"""Returns target index if possible to deduce from belief which target was observed, else returns None."""
		if self.doPrint:
			print "From deduceWhichTarget:"
			print "\tdeduceWhichTarget: " + str(observedState)
			print "\tpartialBeliefs: "
		targetsPossiblyAtObservedState = []
		for i in range(len(self.partialBeliefs)):
			if self.doPrint:
				print str(self.partialBeliefs[i])
			if self.partialBeliefs[i].get(observedState, None) != None:
				targetsPossiblyAtObservedState.append(i)
		if len(targetsPossiblyAtObservedState) == 1:
			return targetsPossiblyAtObservedState[0]
		return None

	def printUpdateResults(self):
		"""Print to terminal the results of most recent update."""
		for i in range(self.numTargets):
			print "Belief target " + str(i+1) + " (sum: " + str(sum(self.partialBeliefs[i].values())) + "):"
			for targetState in self.partialBeliefs[i].keys():
				print "\t" + str(targetState) + ": " + str(self.partialBeliefs[i][targetState])
		#print "Belief (sum: " + str(sum(self.belief.values())) + "): "
		#for key in self.belief.keys():
		#	print str(key) + ": " + str(self.belief[key])
		self.printWorld()
		print "\n".join(self.printList)

	def getNewUnnormalizedPartialBelief(self, partialBelief, observation, agentCompoundEndState, printList = None):
		"""Returns updated unnormalized partial belief."""
		newPartialBelief = dict()
		
		# Calculate new unnormalized belief based on tau (belief transition function)]
		if self.observationFcn.targetStatesFromObservation[observation].get(agentCompoundEndState, None) == None:
			return False
		else:
			possibleTargetEndStates = self.observationFcn.targetStatesFromObservation[observation][agentCompoundEndState]
		
		count = 0
		#for targetEndState in self.robotStates:
		for targetEndState in possibleTargetEndStates:
			p = 0
			for targetState in partialBelief.keys():
				p += self.Ttl(targetState, targetEndState)*partialBelief[targetState]
				count += 1
			newPartialBelief[targetEndState] = self.Ol(agentCompoundEndState, targetEndState, observation)*p

		# If all calculated probabilities are 0, the new belief is invalid
		if sum(newPartialBelief.values()) < 1e-10:#Util.EPSILON:
			return False
		
		# Add info to print list, if applicable
		if printList != None:
			printList.append("Iterations: " + str(count))

		return newPartialBelief


	def getNewPartialBelief(self, partialBelief, observation, agentCompoundEndState, printList = None):
		"""Get a new belief based on previous belief, action and observation."""
		newPartialBelief = self.getNewUnnormalizedPartialBelief(partialBelief, observation, agentCompoundEndState, printList)
		
		# Normalize and remove entries with zero probability
		if newPartialBelief == False:
			return False
		newPartialBelief = self.getNormalizedNonZeroBelief(newPartialBelief)

		return newPartialBelief

	def getNewUnnormalizedBelief(self, belief, action, observation, agentCompoundState, agentCompoundEndState, printList = None):
		"""Returns updated unnormalized belief."""
		newBelief = dict()

		if agentCompoundEndState != self.Ta(agentCompoundState, action):
			print "(s1, a, s2) IMPOSSIBLE!"
			return False
		
		# Calculate new unnormalized belief based on tau (belief transition function)
		if self.observationFcn.targetCompoundStatesFromObservation[observation].get(agentCompoundEndState, None) != None:
			possibleTargetCompoundEndStates = self.observationFcn.targetCompoundStatesFromObservation[observation][agentCompoundEndState]
		else:
			return False # If no targetCompoundStates are possible to end up in, all probabilities are zero (meaning impossible to make obervation given belief and action)
		count = 0
		for targetCompoundEndState in possibleTargetCompoundEndStates:
			endState = (agentCompoundEndState, targetCompoundEndState)
			p = 0.0
			for targetCompoundState in belief.keys():
				state = (agentCompoundState, targetCompoundState)
				p += self.T(state, action, endState)*belief[targetCompoundState]
				count += 1
			newBelief[targetCompoundEndState] = self.O(endState, observation)*p

		# If all calculated probabilities are 0, the new belief is invalid
		if sum(newBelief.values()) < Util.EPSILON:
			return False
		
		# Add info to print list, if applicable
		if printList != None:
			printList.append("Observation corresponds to " + str(len(possibleTargetCompoundEndStates)) + " possible target compound states.")
			printList.append("Iterations: " + str(count))

		return newBelief

	def getNewBelief(self, belief, action, observation, agentCompoundState, agentCompoundEndState, printList = None):
		"""Get a new belief based on previous belief, action and observation."""
		#print "from getNewBelief: belief: " + str(belief)
		#print "from getNewBelief: observation: " + str(observation)
		
		# Get new unnormalized belief, normalize and remove entries with zero probability
		newBelief = self.getNewUnnormalizedBelief(belief, action, observation, agentCompoundState, agentCompoundEndState, printList)
		if newBelief == False:
			return False # Means that it is impossible to make observation given belief and action
		newBelief = self.getNormalizedNonZeroBelief(newBelief)

		return newBelief

	def getNormalizedNonZeroBelief(self, belief = None):
		"""Returns a normalized belief with zero-entries popped."""
		belief = self.belief if belief == None else belief
		normFactor = 1/sum(belief.values())
		for targetCompoundState in belief.keys():
			belief[targetCompoundState] *= normFactor
			if abs(belief[targetCompoundState]) < Util.EPSILON:
				belief.pop(targetCompoundState)
		return belief

	def refineBelief(self, observedTargets, observationList = None):
		"""Refines the belief based on knowledge that observation i corresponds to specified targets."""
		if self.doPrint:
			print "From refineBelief:"
			print "Beliefs before refinement:"
			for i in range(len(self.partialBeliefs)):
				print "Belief target " + str(i + 1) + ":"
				partialBelief = self.partialBeliefs[i]
				for targetState in partialBelief.keys():
					print "\t" + str(targetState) + ": " + str(partialBelief[targetState])
		#print "\tself.ambiguousObservation: " + str(self.ambiguousObservation)
		#print "\tself.observationList: " + str(self.observationList)
		if observationList == None:
			if len(observedTargets) == len(self.observationList):
				observationList = self.observationList
				for i in range(len(observedTargets)):
					if observedTargets[i] == None:
						observedTargets[i] = list()
			elif len(observedTargets) == len(self.ambiguousObservation):
				observationList = self.ambiguousObservation
		#print "\tobservedTargets: " + str(observedTargets)
		#print "\tobservationList: " + str(observationList)
		# Remove impossible beliefs based on observations
		for i in range(len(observationList)):
			observedState = observationList[i]
			for targetId in range(self.numTargets):

				if self.modelRepr == Model.SYSTEM:
					for targetCompoundState in self.belief.keys():
						targetState = targetCompoundState[targetId]
						# Remove all beliefs corresponding to observed target not being in the observed state
						# and all beliefs corresponding to non-observed targets being in the observed state
						observedTargetNotInObservedState = targetId in observedTargets[i] and targetState != observedState
						nonObservedTargetInObservedState = targetId not in observedTargets[i] and targetState == observedState
						if observedTargetNotInObservedState or nonObservedTargetInObservedState:
							#print "Refined belief, removed:"
							#print "\ttargetId+1: " + str(targetId+1)
							#print "\ttargetState: " + str(targetState)
							#print "\tobservedState: " + str(observedState)
							self.belief.pop(targetCompoundState)
				elif self.modelRepr == Model.DECENTRALIZED:
					for targetState in self.partialBeliefs[targetId].keys():
						# Remove all beliefs corresponding to observed target not being in the observed state
						# and all beliefs corresponding to non-observed targets being in the observed state
						observedTargetNotInObservedState = targetId in observedTargets[i] and targetState != observedState
						nonObservedTargetInObservedState = targetId not in observedTargets[i] and targetState == observedState
						if observedTargetNotInObservedState or nonObservedTargetInObservedState:
							#print "Refined belief, removed:"
							#print "\ttargetId+1: " + str(targetId+1)
							#print "\ttargetState: " + str(targetState)
							#print "\tobservedState: " + str(observedState)
							self.partialBeliefs[targetId].pop(targetState)

		# Normalize belief
		if self.modelRepr == Model.SYSTEM:
			self.belief = self.getNormalizedNonZeroBelief()
			self.partialBeliefs = self.getPartialBeliefs()
		elif self.modelRepr == Model.DECENTRALIZED:
			for targetId in range(self.numTargets):
				self.partialBeliefs[targetId] = self.getNormalizedNonZeroBelief(self.partialBeliefs[targetId])
		
		# Print results
		if self.doPrint:
			print "Beliefs after refinement:"
			for i in range(len(self.partialBeliefs)):
				print "Refined partial belief " + str(i + 1)
				partialBelief = self.partialBeliefs[i]
				for targetState in partialBelief.keys():
					print "\t" + str(targetState) + ": " + str(partialBelief[targetState])

	def getPartialBeliefs(self, belief = None):
		"""Returns beliefs for each individual target."""
		belief = self.belief if belief == None else belief
		partialBeliefs = [dict() for i in range(self.numTargets)]
		for targetCompoundState in belief.keys():
			for i in range(self.numTargets):
				targetState = targetCompoundState[i]
				if partialBeliefs[i].get(targetState, None) == None:
					partialBeliefs[i][targetState] = 0
				partialBeliefs[i][targetState] += belief[targetCompoundState]
		return partialBeliefs

	def isHumanNeeded(self):
		"""Returns whether human could be used to resolve belief ambiguity."""
		return len(self.ambiguousObservation) > 0

	def getHumanInput(self, observationList = None):
		"""
		Takes human input about observation-target correspondance.
		Returns a list of either target ids or None at each index of list of observations. (This is what is returned from Unity application)
		"""
		observationList = self.ambiguousObservation if observationList == None else observationList
		print "observationList: " + str(observationList)
		self.printWorld()
		observedTargets = [None for i in range(len(observationList))] # Targets observed in observation i
		validInputs = {str(i + 1) for i in range(self.numTargets)}
		for i in range(len(observationList)):
			observedState = observationList[i]
			humanInput = None
			invalidInput = True
			while invalidInput:
				if humanInput != None:
					print "Invalid input!"
				humanInput = raw_input("Which targets are at " + str(observedState) + " (comma-separated ids)? ")
				invalidInput = False
				for el in humanInput.split(","):
					if el not in validInputs:
						invalidInput = True
						break
			observedTargets[i] = []
			for targetIdStr in humanInput.split(","):
				targetId = int(targetIdStr) - 1 # -1 since starts at 0
				observedTargets[i].append(targetId)
		print "Human info: " + str(observedTargets) + "\n"
		print "\n==============================================================================\n"
		return observedTargets

	def getSimulatedHumanInput(self, observationList = None):
		"""
		Simulates perfect human input about observation-target correspondance.
		Returns a list of either target ids or None at each index of list of observations. (This is what is returned from Unity application)
		"""
		observationList     = self.ambiguousObservation if observationList == None else observationList
		observedTargets     = [None for i in range(len(observationList))] # Targets observed in observation i
		targetCompoundState = self.state[1]
		for i in range(len(observationList)):
			observedState = observationList[i]
			for targetId in range(self.numTargets):
				targetState = targetCompoundState[targetId]
				if targetState == observedState:
					if observedTargets[i] == None:
						observedTargets[i] = []
					observedTargets[i].append(targetId)
		if self.doPrint:
			print "Simulated human gave input: " + str(observedTargets)
		return observedTargets

	def printWorld(self):
		"""Print a 2D grid representation of current state in terminal."""
		self.world.printWorld(self.state)