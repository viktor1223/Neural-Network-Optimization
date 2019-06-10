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

class Progeny():
    def __init__(self, geneSet):
        self.geneSet = geneSet

    def generate_parent(self, length):
        genes = []
        while(len(genes)<length):
            sampleSize=min(length-len(genes), len(self.geneSet))
            genes.extend(random.sample(self.geneSet, sampleSize))
        return ''.join(genes)

    def mutate(self, parent):
        index = random.randrange(0, len(parent))
        childGenes = list(parent)
        newGene, alternate = random.sample(self.geneSet, 2)
        childGenes[index] = alternate \
            if newGene == childGenes[index] \
            else newGene
        return ''.join(childGenes)
