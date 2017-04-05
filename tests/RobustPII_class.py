"""Tests on the Robust-Promethee class."""
import promethee as prom
import data_reader as dr
import random


def test_ranking():
    """Test if the ranking obtained is the same as in Robust PII article."""
    data_set = 'data/HDI/raw.csv'
    alts = dr.open_raw(data_set)[0]
    weights = [0.5, 0.5]
    ceils = [3, 3]
    robust = prom.RobustPII(alts, weights=weights, ceils=ceils,
                            R=10000, m=5)

    rank = robust.ranking
    scores = robust.scores
    for i in range(len(rank)):
        print(str(rank[i]) + '::' + str(scores[rank[i]]))
