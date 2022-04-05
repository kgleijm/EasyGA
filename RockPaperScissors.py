from GeneticAlgorithm import GeneticAlgorithm
import random as r


def RPSsimulation(genome, rounds):

    """
    This is not an actual rock paper scissors simulation as its just an algorithm that learns to pick the winning option
    """

    wins = 0
    possibilities = [1, 1, 1]
    for i in range(rounds):

        opponentChoice = r.choice(range(0, 3))
        observation = [0, 0, 0]
        observation[opponentChoice] = 1
        myChoice = GeneticAlgorithm.HelperFunctions.choose(genome, observation, possibilities)

        if (myChoice - 1) % 3 == opponentChoice:
            wins += 1

    return wins/rounds


def fitnessFunc(genome: list[list[float]]):
    return RPSsimulation(genome, 100)


GA = GeneticAlgorithm("RockPaperScissors", fitnessFunc, 3, 3, mutationChance=0.05)
GA.run()
