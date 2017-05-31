"""Tests on the Robust-Promethee class."""
import promethee as prom
import data_reader as dr
import random


def test_ranking():
    """Test if the ranking obtained is the same as in Robust PII article.

    concerned article:
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
