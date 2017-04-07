"""Genetic algorithm searching for references sets reproducing the PII ranking.

The population consists in a list of references sets (RS) which is evaluated
using kendall's correlation tau.
"""
import random
from scipy import stats
import numpy as np
import bisect
import time

import promethee as prom
import data_reader as dr


def genetic_search(alternatives, seed=None, weights=None, ceils=None,
                   coefficients=None, alt_num=-1, RS_size=4, pop_size=400,
                   mut_prob=0.01, MAXIT=400):
    """Search for references sets reproducing PII."""
    # Initialisation of the PrometheeII, ReferencedPII objects
    promethee = prom.PrometheeII(alternatives, seed=seed, alt_num=alt_num,
                                 ceils=ceils, weights=weights,
                                 coefficients=coefficients)
    prom_ranking = promethee.ranking

    population = initial_population(alternatives, pop_size, RS_size)
    referenced = prom.ReferencedPII(alternatives, seed=seed, alt_num=alt_num,
                                    ceils=ceils, weights=weights,
                                    ref_set=population[0],
                                    coefficients=coefficients)
    evaluations = compute_evaluations(population, prom_ranking, referenced)

    best_score = max(evaluations)
    best_RS_ever = population[evaluations.index(best_score)]

    it = 0
    while(abs(best_score - 1) > 1e-5 and it < MAXIT):
        # print("it:" + str(it) + '  best score:' + str(best_score))
        parents = chose_parents(population, evaluations, pop_size)
        population = combine_parents(parents)
        population = mutate_population(population, mut_prob)
        evaluations = compute_evaluations(population, prom_ranking, referenced)
        if max(evaluations) > best_score:
            best_score = max(evaluations)
            best_RS_ever = population[evaluations.index(best_score)]
        it += 1

    # print(best_score)
    # print(prom_ranking)
    # scores = referenced.compute_scores(ref_set=best_RS_ever)
    # ranking = referenced.compute_ranking(scores)

    # print(stats.kendalltau(ranking, prom_ranking)[0])
    return best_score


def mutate_population(population, p):
    """Mutate each evaluation of each reference with probability p."""
    new_population = []
    for RS in population:
        new_RS = []
        for ref in RS:
            new_ref = ref[:]
            for i in range(len(new_ref)):
                if p >= random.random():
                    new_ref[i] = new_ref[i]*random.uniform(0.5, 1.5)
            new_RS.append(new_ref)
        new_population.append(new_RS)
    return new_population


def combine_parents(parents):
    """Combine the parents to obtain a new population.

    The combination works reference wise.
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


def dominated_rs(RS):
    """Transorm a banal Reference set in the equivalent dominated one.

    This is needed to have meaningfull combinations.
    """
    RS_per_criterion = list(map(list, zip(*RS)))
    RS_per_crit = [sorted(crit) for crit in RS_per_criterion]
    RS_dominated = list(map(list, zip(*RS_per_crit)))
    return RS_dominated


def chose_parents(population, evaluations, pop_size):
    """Chose the parents of our future evaluations.

    The procedure is the following:
        1. translate and normalize the evaluations to decrease the number of
           times bad evaluations will be parents
        2. multiply these score by the population size to obtain the proportion
           of times each individual should be parent. The proportion can be
           seen as an integer part (quantities) and a fractionnal part (rest)
        3. Each individual can be parent only an entire number of times,
           therefore we use the rests to increase some of the "quantities" in
           random way with proportional probabilities.
           The roulette wheele with equally spaced pointers is used to do so.
    """
    translated_evals = [max(s - ((min(evaluations) + max(evaluations))/2), 1e-9)
                        for s in evaluations]

    normalized_evals = [ev/sum(translated_evals) for ev in translated_evals]

    proportions = [(s*pop_size) for s in normalized_evals]

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


def initial_population(alternatives, p_size, RS_size):
    """Compute the initial population."""
    eval_per_criterion = list(map(list, zip(*alternatives)))
    min_per_crit = [min(crit) for crit in eval_per_criterion]
    max_per_crit = [max(crit) for crit in eval_per_criterion]

    pop = []
    for i in range(p_size):
        RS = []
        for j in range(RS_size):
            RS.append([random.uniform(min_per_crit[crit], max_per_crit[crit])
                       for crit in range(len(min_per_crit))])
        pop.append(RS)
    return pop


def compute_evaluations(population, prom_ranking, referenced):
    """Evaluate each individual of the population."""
    population_scores = []
    for RS in population:
        refflows = referenced.compute_scores(ref_set=RS)
        refrank = referenced.compute_ranking(refflows)

        tau = stats.kendalltau(refrank, prom_ranking)[0]
        population_scores.append(tau)
    return population_scores


def save_res_to_file(file_name, alt_num, succes, failures, failures_tau):
    """Print the results in a file."""
    output = open(file_name, 'a')
    output.write('\n')
    title = '#'*5 + ' ' + str(alt_num) + 'alts ' + '#'*5 + '\n'
    output.write(title)
    output.write("succesfull seeds: " + str(succes) + "\n")
    output.write("failed  seeds: " + str(failures) + "\n")
    output.write("failed taus: " + str(failures_tau) + "\n")
    output.write("\n"*3)


if __name__ == '__main__':
    data_sets = ['EPI', 'SHA', 'GEQ']
    data_set = 'data/EPI/raw.csv'
    alts = dr.open_raw(data_set)[0]
    # alts = alts[0:20]
    weights, ceils = None, None
    pseed = range(20)
    alternative_numbers = [20, 25, 30, 40, 50]
    alternative_numbers = [20]

    for data_set in data_sets:
        input_file = 'data/' + str(data_set) + '/raw.csv'
        output = 'res/ReferencedPII_genetic_search/' + str(data_set) + '.txt'
        alts = dr.open_raw(input_file)[0]
        for alt_num in alternative_numbers:
            succes = []
            failures = []
            failures_tau = []
            for s in pseed:
                t1 = time.time()
                tau = genetic_search(alts, seed=s, weights=weights,
                                     ceils=ceils, alt_num=alt_num)
                print(str(s) + ', time: ' + str(time.time() - t1) + ', tau: '
                      + str(tau))
                if (tau > 1 - 1e-5):
                    succes.append(s)
                else:
                    failures.append(s)
                    tau_rounded = int(tau*1000)/1000
                    failures_tau.append(tau_rounded)
            save_res_to_file(output, alt_num, succes, failures, failures_tau)

    print("time :" + str(time.time() - t1))
