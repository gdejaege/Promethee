"""Test file of the new Promethee module and the Promethee class."""

import promethee as prom
import data_reader as dr


def test_ranking():
    """Test that PII computes the same ranking that in the article RobustPII.

    The following mappings should however be applied between countries and and
    indices:
        0 - Norway            10 - Singapore
        1 - Australia         11 - Hong Kong
        2 - Switzerland       12 - Liechtenstein
        3 - Denmark           13 - Sweden
        4 - Netherlands       14 - United Kingdom
        5 - Germany           15 - Iceland
        6 - Ireland           16 - Korea
        7 - United States     17 - Israel
        8 - Canada            18 - Luxembourg
        9 - New Zealand       19 - Japan

    The ranking expected is:
        2::0.31491228070175437
        1::0.2500000000000007
        8::0.18245614035087707
        11::0.18070175438596484
        19::0.16315789473684195
        17::0.16228070175438677
        9::0.059649122807016945
        13::0.058771929824561676
        0::0.04210526315789358
        5::0.007894736842106042
        14::-0.02543859649122777
        16::-0.02807017543859552
        10::-0.07105263157894759
        4::-0.08070175438596594
        18::-0.09824561403508743
        15::-0.13771929824561518
        6::-0.14999999999999925
        3::-0.17631578947368398
        7::-0.28859649122807074
        12::-0.3657894736842105
    """
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
    """Test the function computing the amount of RR between two rankings.

    The rankings compared are :
        * [1, 2, 3, 4, 5, 6]
        * [6, 4, 3, 1, 5]
        there should therefore be 7 rank reversals:
            (6,1);(6,3);(6,4);(6,5);
            (4,3);(4,1);
            (3,1)
    """
    # we don't care about the parameters, we just want to initialise the object
    data_set = 'data/HDI/raw.csv'
    alts = dr.open_raw(data_set)[0]
    coeffs = [0.61224, 1.2]
    weights = [0.5, 0.5]
    promethee = prom.PrometheeII(alts, weights=weights, coefficients=coeffs)

    # Here start the real interresting test
    ranking_init = [1, 2, 3, 4, 5, 6]
    ranking_new = [6, 4, 3, 1, 5]
    alt_removed = 2
    rr = promethee.compare_rankings(ranking_init, ranking_new, alt_removed)
    """Check that the arguments are not modified."""
    print(ranking_init)
    print(ranking_new)
    print(rr)


def test_rr_analysis(data='HDI'):
    """Check that the rank reversals are correct.

    These rank reversal should be compared to the one occuring in the article:
        'About the computation of robust PROMETHEE II rankings: empirical
        evidence' by De Smet.

    The following mappings should however be applied between countries and and
    indices:
        0 - Norway            10 - Singapore
        1 - Australia         11 - Hong Kong
        2 - Switzerland       12 - Liechtenstein
        3 - Denmark           13 - Sweden
        4 - Netherlands       14 - United Kingdom
        5 - Germany           15 - Iceland
        6 - Ireland           16 - Korea
        7 - United States     17 - Israel
        8 - Canada            18 - Luxembourg
        9 - New Zealand       19 - Japan
    """
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

    # print("initial ranking :")
    # print(promethee.ranking)
    # print("initial scores :")
    # print(promethee.scores)
    print("Rank reversals:")
    rr = promethee.compute_rr_number(True)
    print("rank reverasal quantity: " + str(rr))
    rr_instances = promethee.analyse_rr()
    print('rank reversal recap :')
    print(rr_instances)
