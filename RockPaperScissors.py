from GeneticAlgorithm import GeneticAlgorithm


def fitnessFunc(inp: list[list[float]]):
    print(inp)
    return 0.0

GA = GeneticAlgorithm("RockPaperScissors", fitnessFunc, 3, 3, mutationChance=0.05)
