"""Test file of the new Promethee module."""

import promethee as prom
import data_reader as dr


def test_ranking():
    """Test that PII computes the same ranking that in the article RobustPII."""
    data_set = 'data/HDI/raw.csv'
    alts = dr.open_raw(data_set)[0]
    weights = [0.5, 0.5]
    ceils = [3, 3]
    promethee = prom.PrometheeII(alts, weights=weights, ceils=ceils)
    # promethee.scores = promethee.compute_scores()
    scores = promethee.scores
    # promethee.ranking = promethee.compute_ranking(scores)
    rank = promethee.ranking
    for i in range(len(rank)):
        print(str(rank[i]) + '::' + str(scores[rank[i]]))


def test_rr_counting_function():
    """Test the function computing the amount of RR between two rankings."""
    # we don't care about the parameters, we just want to initialise the object
    data_set = 'data/HDI/raw.csv'
    alts = dr.open_raw(data_set)[0]
    coeffs = [0.61224, 1.2]
    weights = [0.5, 0.5]
    promethee = prom.PrometheeII(alts, weights=weights, coefficients=coeffs)
    ranking_init = [1, 2, 3, 4, 5, 6]
    ranking_new = [6, 4, 3, 1, 5]
    alt_removed = 2
    rr = promethee.compare_rankings(ranking_init, ranking_new, alt_removed)
    """Check that the arguments are not modified."""
    print(ranking_init)
    print(ranking_new)


def test_rr_analysis(data='HDI'):
    """Check that the rank reversals are correct."""
    # Data initialisation according to the data set
    if(data == 'HDI'):
        data_set = 'data/HDI/raw.csv'
        alts = dr.open_raw(data_set)[0]
        ceils = [3, 3]
        weights = [0.5, 0.5]
        promethee = prom.PrometheeII(alts, weights=weights, ceils=ceils)

    elif(data == 'SHA'):
        data_set = 'data/SHA/raw_20.csv'
        alts, weights, coeff, ceils = dr.open_raw(data_set)
        promethee = prom.PrometheeII(alts, weights=weights, ceils=ceils)

    elif(data == 'EPI'):
        data_set = 'data/EPI/raw.csv'
        alts = dr.open_raw(data_set)[0]
        alts = alts[0:20]
        seed = 0
        promethee = prom.PrometheeII(alts, seed=seed)

    print("initial ranking :")
    print(promethee.ranking)
    print("initial scores :")
    print(promethee.scores)
    rr = promethee.compute_rr_number(True)
    print(rr)
    rr_instances = promethee.analyse_rr()
    print(rr_instances)
