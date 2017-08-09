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