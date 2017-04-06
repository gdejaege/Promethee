"""Test file of the ReferencedPII class."""

import promethee as prom
import data_reader as dr
import random

def compare_refflows(refnumber):
    random.seed()
    seed = random.randint(1, 1000)
    print(seed)
    ceils = [3, 3, 3]

    a1 = [4, 8, 9]
    a2 = [7, 5, 4]
    a3 = [4.5, 10, 8]
    a4 = [7.4, 6, 9]
    
    A = [a1, a2, a3, a4]
    RS = [[random.random()*10 for i in range(3)] for j in range(refnumber)]

    referenced = prom.ReferencedPII(A, ref_set=RS, ceils=ceils, seed=seed)
    ref_scores = referenced.scores
    for i, alt in enumerate(A):
        RS_alt = RS[:]
        RS_alt.append(alt)
        promethee = prom.PrometheeII(RS_alt, ceils=ceils, seed=seed)
        scores = promethee.scores
        if abs(scores[-1] - ref_scores[i]) < 1e-5:
            print("ok")
        else:
            print("There is something wrong")
            print(scores)

