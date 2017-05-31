"""Test file of the ReferencedPII class."""

import promethee as prom
import data_reader as dr
import random


def compare_refflows():
    """Check if the ref-flow computed with ReferencedPII object is correct."""
    data_set = 'HDI'
    random.seed()
    seed = random.randint(1, 1000)
    print(seed)
    alt_num = 20
    ref_number = 4
    strategy = prom.strategy2

    input_file = 'data/' + str(data_set) + '/raw.csv'
    alternatives = dr.open_raw(input_file)[0]

    referenced = prom.ReferencedPII(alternatives, strategy=strategy, seed=seed)

    SRP = referenced.SRP
    ref_scores = referenced.scores
    for i, alt in enumerate(alternatives):
        SRP_alt = SRP[:]
        SRP_alt.append(alt)
        promethee = prom.PrometheeII(SRP_alt, seed=seed)
        scores = promethee.scores
        if abs(scores[-1] - ref_scores[i]) < 1e-5:
            print("ok")
        else:
            print("There is something wrong")
            print(scores)
