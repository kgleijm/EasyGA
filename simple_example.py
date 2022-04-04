
from GeneticAlgorithm import GeneticAlgorithm

"""
This simple demonstration takes a genome of 100 genes to 2 options
All the genes get added to a single pair of values
the first of the pair gets subtracted from the first an devided by the amount of genes
this leaves us with a float representing fitness
"""


def fitnessFunc(inp: list[list[float]]):
    result = GeneticAlgorithm.InternalFunctions.getSumOfOptions(inp)
    fitness = result[1] - result[0]/len(inp)
    return fitness

GA = GeneticAlgorithm("Development", fitnessFunc, 100, 2, mutationChance=0.05)