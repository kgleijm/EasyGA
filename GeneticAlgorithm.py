import copy
import os.path
from abc import ABC, abstractmethod
import random as r
from typing import Callable, Tuple
import sqlite3


class Individual:

    class Gen:
        def __init__(self, amountOfOptions):
            self.amountOfOptions = amountOfOptions
            self.options = [r.random() for _ in range(self.amountOfOptions)]

        def mutate(self):
            self.options = [r.random() for _ in range(self.amountOfOptions)]



    def __init__(self, amountOfGenes, amountOfOptions):
        self.genome = []
        self.fitness = 0.001
        self.generation = 0
        self.amountOfGenes = amountOfGenes
        self.amountOfOptions = amountOfOptions

        self.genome = [Individual.Gen(amountOfOptions) for _ in range(amountOfGenes)]


    def mutate(self, chance):
        for gen in self.genome:
            if r.random() < chance:
                gen.mutate()


class GeneticAlgorithm:

    def __init__(self, experimentName: str,
                 fitnessDeterminationFunction: Callable[[list[list[float]]], float],
                 amountOfDataInputOptions,
                 amountOfOutcomeOptions,
                 populationSize=10,
                 crossoverChance=0.8,
                 crossOverGenPercentage=0.5,
                 mutationChance=0.02,
                 elitism=2):
        """Constructor takes an experiment name for use in data management
        And a fitness determination function that should provide the fitness of a certain genome.
        """

        self.experimentName = experimentName
        self.fitnessDeterminationFunction = fitnessDeterminationFunction
        self.populationSize = populationSize
        self.amountOfDataInputs = amountOfDataInputOptions
        self.amountOfOutcomeOptions = amountOfOutcomeOptions
        self.population = []
        self.running = True
        self.crossoverChance = crossoverChance
        self.crossOverGenPercentage = crossOverGenPercentage
        self.mutationChance = mutationChance
        self.elitism = elitism

        # Test essential functioning
        GeneticAlgorithm.DataManagement.Test(experimentName)

        self.initialSetup()

        self.run()

    def tick(self):
        """Function containing core logic of the genetic algorithm"""
        # Sort old population and initialize empty new population
        self.population.sort(key=lambda i: i.fitness, reverse=True)
        newPopulation = []


        print("len(population):", len(self.population))

        elites = []
        # Perform elitism
        for i in range(self.elitism):
            newPopulation.append(self.population[i])
            elites.append(self.population[i])

        # Cross over Individuals
        while len(newPopulation) < self.populationSize:
            sample = GeneticAlgorithm.HelperFunctions.pickWeightedSampleByFitness(self.population, 2)
            crossedOverSample = self.crossOverIndividualsByChance(sample, self.crossoverChance, self.crossOverGenPercentage)
            for i in crossedOverSample:
                if len(newPopulation) < self.populationSize:
                    newPopulation.append(i)

        # Mutate individual
        for i in newPopulation:
            if i not in elites:
                i.mutate(self.mutationChance)

        # Determine fitness of Individuals
        self.determineFitnessForIndividuals(newPopulation)



    def crossOverIndividualsByChance(self, parents: list[Individual, Individual], chance: float, percentageOfGenes: float) -> list[Individual, Individual]:
        childA = copy.deepcopy(parents[0])
        childB = copy.deepcopy(parents[1])

        if r.random() < chance:
            for i in range(len(childA.genome)):
                if r.random() < percentageOfGenes:
                    tempGene = childA.genome[i]
                    childA.genome[i] = childB.genome[i]
                    childB.genome[i] = tempGene

        return [childA, childB]

    def mutateIndividuals(self, individuals: list[Individual], chance: float):
        for individual in individuals:
            individual.mutate(chance)

    def determineFitnessForIndividuals(self, individuals: list[Individual]):
        for individual in individuals:
            individual.fitness = self.fitnessDeterminationFunction(GeneticAlgorithm.HelperFunctions.convertGenomeToListOfFloats(individual.genome))


    def run(self):
        while self.running:
            self.tick()

    def initialSetup(self):
        """Sets up initial population pulling valuable genomes from the database or inserting random ones if"""
        initialPopulation = []

        existingFitPopulation = self.DataManagement.queryNBest(self.experimentName, self.populationSize)
        print(f"database query of the algorithm for {self.experimentName} found {len(existingFitPopulation)} of {self.populationSize} Individuals")

        # Fill initial population with best individuals from database
        for rawIndividual in existingFitPopulation:
            newIndividual = Individual(amountOfGenes=self.amountOfDataInputs, amountOfOptions=self.amountOfOutcomeOptions)
            newIndividual.generation = rawIndividual[0]
            newIndividual.fitness = rawIndividual[1]
            newIndividual.genome = rawIndividual[2].split(",")
            initialPopulation.append(newIndividual)

        # Top up initial population if not enough Individuals could be pulled from the database
        while len(initialPopulation) < self.populationSize:
            initialPopulation.append(Individual(amountOfGenes=self.amountOfDataInputs, amountOfOptions=self.amountOfOutcomeOptions))

        self.population = initialPopulation
        self.population.sort(key=lambda i: i.fitness, reverse=True)

        print(f"initial setup added {self.populationSize - len(initialPopulation)} of {self.populationSize} Individuals")

    class InternalFunctions:

        @staticmethod
        def mutateIndividual(self, individual: Individual):
            pass

        @staticmethod
        def crossOverIndividuals(self, individuals: Tuple[Individual, Individual]):
            pass

        @staticmethod
        def determineFitness(fitnessDeterminationFunction: Callable[[list[Individual.Gen]], float], individual: Individual):
            """Function applies the fitness determination function to the genome and saves the result in the designated places"""
            individual.fitness = fitnessDeterminationFunction(individual.genome)

    class DataManagement:

        @staticmethod
        def getQueryResult(experimentName: str, sql: str):
            path = GeneticAlgorithm.DataManagement.getDBPath(experimentName)
            res = None
            with sqlite3.connect(path) as con:
                cur = con.cursor()
                cur.execute(sql)
                res = cur.fetchall()
                cur.close()
            return res

        @staticmethod
        def initializeDBIfNotExists(experimentName: str):
            """Creates directory, database and tables if necessary"""
            # Create path
            dbDirPath = os.path.join(*[os.getcwd(), "GA_DATA"])
            if not os.path.exists(dbDirPath):
                os.mkdir(dbDirPath)

            # Create tables
            path = os.path.join(*[os.getcwd(), "GA_DATA", f"{experimentName}.db"])
            sql = "CREATE TABLE IF NOT EXISTS genomes (" \
                  "ID INTEGER," \
                  "generation INTEGER," \
                  "fitness FLOAT," \
                  "genome varchar" \
                  ");"
            with sqlite3.connect(path) as con:
                cur = con.cursor()
                cur.execute(sql)
                cur.close()

        @staticmethod
        def getDBPath(experimentName: str):
            """Returns path to db file corresponding to experiment name"""
            GeneticAlgorithm.DataManagement.initializeDBIfNotExists(experimentName)
            return os.path.join(*[os.getcwd(), "GA_DATA", f"{experimentName}.db"])

        @staticmethod
        def Test(experimentName: str):
            """Test the database connection"""
            path = GeneticAlgorithm.DataManagement.getDBPath(experimentName)
            print("trying to open database at path:", path)
            with sqlite3.connect(path) as con:
                print(sqlite3.version)
                print("DataManagement Test successful")

        @staticmethod
        def saveGenomeIfValuable(experimentName: str, genome: list[float]):
            """Checks if genome has a higher fitness than all know genomes and saves it if true"""
            # TODO IMPLEMENT
            pass

        @staticmethod
        def queryNBest(experimentName: str, n: int):
            """Returns n genomes with the highest fitness.
            May return any number from 0 till n results ordered by fitness"""

            sql = f"""SELECT * FROM genomes
            ORDER BY fitness DESC
            LIMIT {n}"""

            return GeneticAlgorithm.DataManagement.getQueryResult(experimentName, sql)

    class HelperFunctions:

        @staticmethod
        def pickWeightedSampleByFitness(population: list[Individual], n=2) -> list[Individual]:
            """Returns a list of n individuals chosen by a weighted random"""
            fitnessList = [individual.fitness for individual in population]
            individual_cumulative = [(population[i], sum(fitnessList[:i + 1])) for i in range(len(fitnessList))]
            output = []
            while len(output) < n:
                seed = r.random() * sum(fitnessList)
                for e in individual_cumulative:
                    if seed < e[1]:
                        if e[0] not in output:
                            output.append(e[0])
                            break
            return output

        @staticmethod
        def convertGenomeToListOfFloats(genome: list[Individual.Gen]):
            return [gen.options for gen in genome]

        @staticmethod
        def getSumOfOptions(options: list[list[float]]):
            res = [0]*len(options)
            for option in options:
                for i in range(len(option)):
                    res[i] += option[i]
            return res


        # TODO determine if function is feasible
        # @staticmethod
        # def choose(self, data):
        #     getGenSumOfIndex = lambda index: sum([gen.option[index] for gen in self.genome])
        #     genomeTotalForOption = [getGenSumOfIndex(index) for index in range(len(self.genome))]


def fitnessFunc(inp: list[list[float]]):
    result = GeneticAlgorithm.HelperFunctions.getSumOfOptions(inp)
    fitness = result[1] - result[0]
    return fitness

GA = GeneticAlgorithm("Development", fitnessFunc, 3, 2)
