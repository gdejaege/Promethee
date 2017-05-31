"""Analyse rank reversals occurences in the Robust Promethee method."""
import promethee as prom
import data_reader as dr
import random
import numpy as np


def analyse_rr(data='SHA', max_rep=20, R_parameter=None, m_parameter=None):
    """Analyse the rank reversals occuring in RobustPII."""
    if (data == 'HDI'):
        print('try with another dataset')
        exit()

    elif(data == 'SHA'):
        R = 5000
        m = 9
        # Do not change these parameters ! They are not saved
        data_set = 'data/SHA/raw_20.csv'
        alts = dr.open_raw(data_set)[0]
        weights = [0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667]
        ceils = [17.100, 23.7750, 26.100, 27.3750, 17.9250, 13.5750]
        seed = 1
    else:
        data = 'EPI'
        R = 5000
        m = 16
        # Do not change these parameters ! They are not saved
        data_set = 'data/EPI/raw.csv'
        alts = dr.open_raw(data_set)[0]
        alts = alts[0:20]
        weights, ceils = None, None
        seed = 0

    if R_parameter is not None:
        R = R_parameter
    if m_parameter is not None:
        m = m_parameter

    output = 'res/RobustPII/analyse_rank_reversals/' + str(data) + '.txt'

    promethee = prom.PrometheeII(alts, weights=weights, ceils=ceils, seed=seed)
    promethee_rr_instances = promethee.analyse_rr()

    all_rr_instances = dict()
    for repetition in range(max_rep):
        robust = prom.RobustPII(alts, weights=weights, ceils=ceils, seed=seed,
                                R=R, m=m)
        rr_instances = robust.analyse_rr()
        for key in rr_instances:
            all_rr_instances[key] = \
                    all_rr_instances.get(key, 0) + rr_instances.get(key)

    all_info = []

    key_set = set(all_rr_instances.keys()) | set(promethee_rr_instances.keys())
    for key in key_set:
        line = [key[0], key[1], all_rr_instances.get(key, 0)/max_rep,
                promethee_rr_instances.get(key, 0),
                abs(promethee.scores[key[0]] - promethee.scores[key[1]]),
                abs(robust.scores[key[0]] - robust.scores[key[1]])]
        all_info.append(line)
    print_to_file(output, all_info, promethee.scores, robust.scores, max_rep,
                  R, m)


def print_to_file(file_name, rr_info, PII_scores, Rob_scores, max_rep, R, m):
    """Save the results of the investigations on the RR occurences into a file.

    This should in the end print a beautifull csv file but for the moment it
    just saves everything
    """
    output = open(file_name, 'a')
    # Parameters
    rr_template = "{:3d},{:3d},{:2f},{:2f},{:.5f},{:.5f}"
    output.write("R: " + str(R) + "\n")
    output.write("m: " + str(m) + "\n")
    output.write(str(max_rep) + " repetitions \n")
    # Rank reversals info
    output.write(" a1  a2  qty-rob qty-PII   PII      RobPII : \n")
    for row in rr_info:
        output.write(rr_template.format(*row))
        output.write("\n")
    output.write("\n"*2)

    # Rankings info
    for i in range(len(PII_scores)):
        output.write(str(i) + ':: netflow = ' + str(PII_scores[i])
                     + ':: robflow =' + str(Rob_scores[i]) + '\n')

    std_PII = np.std(PII_scores)
    output.write("standard deviation promethee :" + str(std_PII) + "\n")
    std_Rob = np.std(Rob_scores)
    output.write("standard deviation robust promethee :" + str(std_Rob) + "\n")

    PII_ranking = sorted(range(len(PII_scores)), key=lambda k: PII_scores[k],
                         reverse=True)
    output.write("Promethee Ranking :" + str(PII_ranking) + "\n")
    Rob_ranking = sorted(range(len(PII_scores)), key=lambda k: Rob_scores[k],
                         reverse=True)
    output.write("Robust Ranking :" + str(Rob_ranking) + "\n")
