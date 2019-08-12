"""
@author Viktor Ciroski
"""


"""
Genetic Algorithm Layout

Population
Fitness Calculation
Matting Pool
Parent Seleciton
Matting
    Crossover
    Mutation
Offspring
"""
import random
import datetime
from operator import itemgetter
import random

class Progeny():
    def __init__(self, geneSet, mutation, crossover, keep):
        self.geneSet = geneSet
        self.mutation = mutation
        self.crossover_chance = crossover
        self.keep_rate = keep

    def generate_parent(self, length):
        genes = []
        while(len(genes)<length):
            sampleSize=min(length-len(genes), len(self.geneSet))
            genes.extend(random.sample(self.geneSet, sampleSize))
        return ''.join(genes)

    def mutate(self, parent):

        newGene = ''
        for i in range(len(parent)):
            gene =  random.sample(self.geneSet, 1)

            newGene += gene[0]



        return newGene

    def crossover(self, father, mother):

        netwrok_bit_size = len(father)
        #print(netwrok_bit_size)
        #print(mother)
        #print("Mother Len", len(mother))
        #print(father)
        #print("Father len", len(father))
        baby = ''
        i = 0
        while len(baby) <= netwrok_bit_size:

            probability = random.uniform(0,1)
            if  probability < self.crossover_chance:
                baby += (father[i])
                i += 1
            elif probability < (2*self.crossover_chance):
                baby += (mother[i])
                i += 1
            else:
                gene = [random.randint(0, 1)]
                mutated_gene = (self.mutate(gene))
                baby += (mutated_gene)





        return baby


    def population(self, count, length):
        pop = []
        for _ in range(0, count):
            network = self.generate_parent(length)
            pop.append(network)

        return pop

    @staticmethod
    def fitness(network):
        """Return the accuracy, which is our fitness function."""
        return network.accuracy

    def Offspring(self, population):

        size_of_pop = len(population)

        #sort by most accurate to least
        graded = sorted(population, key=itemgetter(1), reverse=True)


        best_acc = graded[0][1]


        #keep N parents
        keep = int(size_of_pop*self.keep_rate)

        if keep == 0:
            keep = 1

        graded_binary = []
        for i in range(len(graded)):
            graded_binary.append(graded[i][0])
        #print(graded_binary)

        graded_best = graded_binary[:keep]

        #print(graded_best)






        #randomly keep some less accurate parents for diversity
        retain_non_best = int((size_of_pop - keep) * self.keep_rate)
        for _ in range(random.randint(0, retain_non_best)):
            graded_best.append(random.choice(graded_binary[keep:]))

        #fill remaining spots with next evolution

        counter = 0
        #print(size_of_pop)
        #print("inital")

        while len(graded_best) < size_of_pop:
                #print("Coutner", counter)
                counter += 1
                baby = None
                if len(graded_best) == 1:
                     baby = self.mutate(graded_best[0])

                #print(graded_best)
                idx_1 = random.randint(0, len(graded_best)-1)
                idx_2 = random.randint(0, len(graded_best)-1)
                papi = graded_best[idx_1]
                daddy = graded_best[idx_2]



                if daddy != papi:
                    baby = self.crossover(papi, daddy)



                if baby != None:
                    graded_best.append(baby)



        #print(graded_best)
        return graded_best, best_acc
