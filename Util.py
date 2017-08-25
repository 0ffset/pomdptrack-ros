import itertools

EPSILON = 1e-15

def cartesianPower2(S, n):
	"""Returns the n-ary Cartesian power of set S, represented as a list."""
	return list(__cartesianPower(list(S), n))
def __cartesianPower(S, n):
	ntuple = tuple()
	if n == 1:
		for el in S:
			ntuple += ((el,),)
		return ntuple

	for el in S:
		for subS in __cartesianPower(S, n-1):
			ntuple += ((el,) + subS,)

	return ntuple

def cartesianPower(S, n):
	return cartesianProduct(*[list(S)]*n)

def cartesianProduct(*sets):
	return list(itertools.product(*sets))

def cartesianProduct2(*sets):
	return list(__cartesianProduct(*sets))
def __cartesianProduct2(*sets):
	if not sets:
		return tuple(((),))
	tuples = tuple()
	for item in sets[-1]:
		for items in cartesianProduct(*sets[:-1]):
			tuples += (items + (item,),)
	return tuples

def getOrderedCombinationsAsListOfLists(elements, n):
	combinations = []
	if n == 1:
		for el in elements:
			combinations.append([el])
		return combinations

	for el in elements:
		for subElements in getOrderedCombinationsAsListOfLists(elements, n-1):
			combinations.append([el] + subElements)

	return combinations

def getOrderedCombinations(elements, n):
	"""Returns all ordered n-combinations of a list of elements as a list of tuples."""
	combinations = getOrderedCombinationsAsListOfLists(elements, n)
	for i in range(len(combinations)):
		combinations[i] = tuple(combinations[i])
	return combinations

def getCombinationsDifferentSets(sets):
	"""Returns all combinations where different sets are specified for each element."""
	combinations = []
	__getCombinationsDifferentSets(sets, tuple(), combinations)
	return combinations
def __getCombinationsDifferentSets(sets, combination = tuple(), combinations = []):
	level = len(combination)
	if level == len(sets) - 1:
		for i in sets[level]:
			combinations.append(combination + (i,))
	else:
		for i in sets[level]:
			__getCombinationsDifferentSets(sets, combination + (i,), combinations)