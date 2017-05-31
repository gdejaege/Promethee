"""Genetic algorithm searching for references sets reproducing the PII ranking.

The population consists in a list of sets of references profiles (SRP) which are
evaluated using kendall's correlation tau.
"""
import random
from scipy import stats
import numpy as np
import bisect
import time

import promethee as prom
import data_reader as dr


def genetic_search(alternatives, seed=None, weights=None, ceils=None,
                   coefficients=None, alt_num=-1, SRP_size=4, pop_size=600,
                   mut_prob=0.01, MAXIT=50):
    """Search for references sets reproducing PII with a genetic algorithm.

    Inputs:
        alternatives - matrix composed of one list of evaluations for each
                       alternative.

        seed - seed provided to python pseudo random number generator. It is
               used to create some random (w, F) for the method if these are not
               provided as arguments. See promethee.py to see how this is done

        weights - list of the relative importance (or weigths) of all criteria.

        ceils - list of the values of the strict preference thresholds for all
                criteria (p).

        coefficients - if 'ceils' is not provided, some new ceils will be
                       computed as these coefficents time the amplitude
                       between the highest and lowest evaluation of each
                       criterion.

        alt_num - quantity of alternatives from 'alternative' which must be
                  kept.

        SRP_size - quantity of reference profiles searched.

        pop_size - size of the population.

        mut_prob - probability of mutation of each of the evaluation of each
                   individual.

        MAXIT - maximal number of iterations of the procedure.
    """
    # Initialisation of the PrometheeII, ReferencedPII objects
    promethee = prom.PrometheeII(alternatives, seed=seed, alt_num=alt_num,
                                 ceils=ceils, weights=weights,
                                 coefficients=coefficients)
    prom_ranking = promethee.ranking
    random.seed()

    population = initial_population(alternatives, pop_size, SRP_size)
    referenced = prom.ReferencedPII(alternatives, seed=seed, alt_num=alt_num,
                                    ceils=ceils, weights=weights,
                                    ref_set=population[0],
                                    coefficients=coefficients)

    evaluations = compute_evaluations(population, prom_ranking, referenced)

    best_score = max(evaluations)
    best_SRP_ever = population[evaluations.index(best_score)]

    it = 0
    while(abs(best_score - 1) > 1e-5 and it < MAXIT):
        # print("it:" + str(it) + '  best score:' + str(best_score))
        parents = chose_parents(population, evaluations, pop_size)
        population = combine_parents(parents)
        population = mutate_population(population, mut_prob)
        evaluations = compute_evaluations(population, prom_ranking, referenced)
        if max(evaluations) > best_score:
            best_score = max(evaluations)
            best_SRP_ever = population[evaluations.index(best_score)]
        it += 1

    return best_score


def mutate_population(population, p):
    """Mutate each evaluation of each profile with probability p."""
    new_population = []
    for SRP in population:
        new_SRP = []
        for ref in SRP:
            new_ref = ref[:]
            for i in range(len(new_ref)):
                if p >= random.random():
                    new_ref[i] = new_ref[i]*random.uniform(0.75, 1.25)
            new_SRP.append(new_ref)
        new_population.append(new_SRP)
    return new_population


def combine_parents(parents):
    """Combine the parents to obtain a new population.

    The combination works profile wise. This means that each reference serves
    as one chomosome.
    """
    new_population = []
    random.shuffle(parents)
    for i in range(len(parents)//2):
        parent1 = parents.pop()
        parent1 = dominated_rs(parent1)
        parent2 = parents.pop()
        parent2 = dominated_rs(parent2)
        child1 = []
        child2 = []

        for j in range(len(parent1)):
            if random.randint(0, 1):
                child1.append(parent1[j])
                child2.append(parent2[j])
            else:
                child1.append(parent2[j])
                child2.append(parent1[j])
        new_population.append(child1)
        new_population.append(child2)
    return new_population


def dominated_rs(SRP):
    """Transorm a banal Reference set in the equivalent dominated one.

    This is needed to have meaningfull combinations.
    """
    SRP_per_criterion = list(map(list, zip(*SRP)))
    SRP_per_crit = [sorted(crit) for crit in SRP_per_criterion]
    SRP_dominated = list(map(list, zip(*SRP_per_crit)))
    return SRP_dominated


def chose_parents(population, evaluations, pop_size):
    """Chose the parents of our future individuals.

    Input:
        population - actual population, list of SRP.
        evaluations - list of evaluation of each of the individuals.
        pop_size - size of the population.

    The procedure is the following:
        1. translate and normalise the evaluations to decrease the number of
           times bad evaluations will be parents

        2. multiply these score by the population size to obtain the proportion
           of times each individual should be a parent. The proportion can be
           seen as an integer part (quantities) and a fractionnal part (rest)

        3. Each individual can be parent only an entire number of times,
           therefore we use the rests to increase some of the "quantities" in
           random way with proportional probabilities.
           The roulette wheele with equally spaced pointers is used to do so.
    """
    # 1. Translate and normalize
    translated_evals = [max(s - ((min(evaluations) + max(evaluations))/2), 1e-9)
                        for s in evaluations]

    normalized_evals = [ev/sum(translated_evals) for ev in translated_evals]

    # 2. Fractional proportion of time each inividual will be parent
    proportions = [(s*pop_size) for s in normalized_evals]

    # 2. quantities = integer part
    quantities = [int(s) for s in proportions]
    rest = [proportions[i] - quantities[i] for i in range(len(proportions))]
    cumulative_rest = [rest[0]]
    for i in range(1, len(rest)):
        cumulative_rest.append(cumulative_rest[i-1] + rest[i])

    # number of parents still missing in quantities
    missing = pop_size - sum(quantities)
    # equally space pointers (the last must be poped because of the % operation)
    pointers = list(np.linspace(0, cumulative_rest[-1], missing + 1))
    pointers.pop()
    offset = random.uniform(0, cumulative_rest[-1])
    pointers = [(p + offset) % cumulative_rest[-1] for p in pointers]
    for p in pointers:
        index = bisect.bisect(cumulative_rest, p)
        quantities[index] += 1

    parents = []
    for i in range(len(quantities)):
        for j in range(quantities[i]):
            parents.append(population[i])

    return parents


def initial_population(alternatives, p_size, SRP_size):
    """Compute the initial population."""
    eval_per_criterion = list(map(list, zip(*alternatives)))
    min_per_crit = [min(crit) for crit in eval_per_criterion]
    max_per_crit = [max(crit) for crit in eval_per_criterion]

    pop = []
    for i in range(p_size):
        SRP = []
        for j in range(SRP_size):
            SRP.append([random.uniform(min_per_crit[crit], max_per_crit[crit])
                       for crit in range(len(min_per_crit))])
        pop.append(SRP)
    return pop


def compute_evaluations(population, prom_ranking, referenced):
    """Evaluate each individual of the population.

    Input:
        population - all SRP considered at this iteration.
        prom_ranking - ranking obtained with the PrometheeII method.
        referenced - object representing the Referenced Promethee method,
                     it is used to compute a referenced ranking given an SRP.
    """
    population_scores = []
    for SRP in population:
        refflows = referenced.compute_scores(ref_set=SRP)
        refrank = referenced.compute_ranking(refflows)

        tau = stats.kendalltau(refrank, prom_ranking)[0]
        population_scores.append(tau)
    return population_scores
