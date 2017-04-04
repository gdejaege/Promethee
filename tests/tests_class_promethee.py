"""Test file of the new Promethee module."""
import promethee as prom
import data_reader as dr


def test1():
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


def test2():
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


def test3():
    """Check that the number of rank reversals are the same than in the article.

    (robust promethee with HDI data set)
    """
    data_set = 'data/HDI/raw.csv'
    alts = dr.open_raw(data_set)[0]
    ceils = [3, 3]
    weights = [0.5, 0.5]
    promethee = prom.PrometheeII(alts, weights=weights, ceils=ceils)
    print("initial ranking :")
    print(promethee.ranking)
    print("initial scores :")
    print(promethee.scores)
    rr = promethee.compute_rr_number(True)
    print(rr)


def test4():
    """Check that the number of rank reversals are the same than with matlab.

    (robust promethee with HDI data set)
    """
    data_set = 'data/SHA/raw_20.csv'
    alts, weights, coeff, ceils = dr.open_raw(data_set)
    print(weights)
    print(ceils)
    # ceils = [3, 3]
    # weights = [0.5, 0.5]
    promethee = prom.PrometheeII(alts, weights=weights, ceils=ceils)
    rr = promethee.compute_rr_number(True)
    print(rr)


def test5():
    """Test the number of rank reversals in the 20 first EPI alternatives."""
    data_set = 'data/EPI/raw.csv'
    alts = dr.open_raw(data_set)[0]
    alts = alts[0:20]
    seed = 0

    promethee = prom.PrometheeII(alts, seed=seed)
    rr = promethee.compute_rr_number()
    print(rr)
