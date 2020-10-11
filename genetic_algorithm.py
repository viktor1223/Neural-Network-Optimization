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
        self.length = None

    def generate_parent(self, length):
        self.length = length
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

        #netwrok_bit_size = len(father)
        #print(netwrok_bit_size)
        #print(mother)
        print("\n\n------------------------------------------------\n")
        print("Mother Len", len(mother))
        #print(father)
        print("Father len", len(father))
        print("OG len", self.length)
        
        baby = ''
        i = 0
        #Uniform Crossover operation between both parents 
        while len(baby) < self.length:

            probability = random.uniform(0,1)
            if  probability < self.crossover_chance:
                if father[i] == 1 or father[i] == 0:
                    print("adding FATHER gene")
                    baby += (father[i])
                    i += 1
            elif probability < (2*self.crossover_chance):
                if mother[i] == 1 or mother[i] == 0:
                    print("adding MOTHER gene")
                    baby += (mother[i])
                    i += 1
            else:
                #mutation Probability
                gene = [random.randint(0, 1)]
                mutatedProbability = random.uniform(0,1) 
                if mutatedProbability > self.mutation:
                    mutated_gene = (self.mutate(gene))
                    if mutated_gene == 1 or mutated_gene ==0:
                        print("MUTATING a gene")
                        baby += (mutated_gene)
                        i += 1
                else:
                    if gene[0] == 1 or gene[0] == 0: 
                        print("adding RANDOM gene")
                        baby += str(gene[0])
                        i += 1
            print("BABY GENOME")
            print(baby)

 
        print("Baby len", len(baby))
        print("\n\n------------------------------------------------\n")

        return baby


    def population(self, count, length):
        pop = []
        for _ in range(0, count):
            network = self.generate_parent(length)
            pop.append(network)

        return pop


    def fitness(network):
        """Return the accuracy, which is our fitness function."""
        return network.f1

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
