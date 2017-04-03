"""First test file of the Robust-Promethee method in the Promethee module."""
import promethee as prom
import data_reader as dr
import random


def test1():
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


def test2(max_rep=20):
    """Test the number of rank reversals for different R and m.

    This test is done for the 20 first alternatives of the HDI data set.
    Input:
        max_rep=number of repetitions of the method
    """
    output = 'res/class_RobustPII/test2.txt'
    R_list = [500, 1000, 5000, 10000]
    # R_list = [500, 1000]
    m_list = [3, 5, 6, 7, 8, 10, 15]
    data_set = 'data/HDI/raw.csv'
    alts = dr.open_raw(data_set)[0]
    weights = [0.5, 0.5]
    ceils = [3, 3]

    promethee = prom.PrometheeII(alts, weights=weights, ceils=ceils)
    rr_promethee = promethee.compute_rr_number()

    rr_matrix = []
    for R in R_list:
        rr_row = []
        for m in m_list:
            rr = 0
            for repetition in range(max_rep):
                random.seed()
                robust = prom.RobustPII(alts, weights=weights, ceils=ceils,
                                        R=R, m=m)
                rr += robust.compute_rr_number()
            rr = rr/max_rep
            rr_row.append(rr)
        print(rr_row)
        rr_matrix.append(rr_row)
    print_rr_to_file(output, rr_matrix, R_list, m_list, rr_promethee)


def test3(max_rep=20):
    """Test the number of rank reversals for different R and m.

    This test is done for the 20 first alternatives of the SHA data set.
    Input:
        max_rep=number of repetitions of the method
    """
    output = 'res/class_RobustPII/test3.txt'
    R_list = [1000, 4000, 7000, 12000]
    m_list = [4, 6, 8, 9, 12]
    data_set = 'data/SHA/raw_20.csv'
    alts = dr.open_raw(data_set)[0]
    weights = [0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667]
    ceils = [17.100, 23.7750, 26.100, 27.3750, 17.9250, 13.5750]

    promethee = prom.PrometheeII(alts, weights=weights, ceils=ceils)
    rr_promethee = promethee.compute_rr_number()

    rr_matrix = []
    for R in R_list:
        rr_row = []
        for m in m_list:
            rr = 0
            for repetition in range(max_rep):
                random.seed()
                robust = prom.RobustPII(alts, weights=weights, ceils=ceils,
                                        R=R, m=m)
                rr += robust.compute_rr_number()
            rr = rr/max_rep
            rr_row.append(rr)
        print(rr_row)
        rr_matrix.append(rr_row)
    print_rr_to_file(output, rr_matrix, R_list, m_list, rr_promethee)


def test4(max_rep=10):
    """Test the number of rank reversals for different R and m.

    This test is done for the 20 first alternatives of the SHA data set.
    Input:
        max_rep=number of repetitions of the method
    """
    output = 'res/class_RobustPII/test4.txt'
    R_list = [500, 1000, 5000, 8000]
    m_list = [4, 6, 8, 9, 12]
    data_set = 'data/EPI/raw.csv'
    alts = dr.open_raw(data_set)[0]
    alts = alts[0:20]
    seed = 0

    promethee = prom.PrometheeII(alts, seed=seed)
    rr_promethee = promethee.compute_rr_number()

    rr_matrix = []
    for R in R_list:
        rr_row = []
        for m in m_list:
            rr = 0
            for repetition in range(max_rep):
                random.seed()
                robust = prom.RobustPII(alts, seed=0, alt_num=20,
                                        R=R, m=m)
                rr += robust.compute_rr_number()
            rr = rr/max_rep
            rr_row.append(rr)
        print(rr_row)
        rr_matrix.append(rr_row)
    print_rr_to_file(output, rr_matrix, R_list, m_list, rr_promethee)


def print_rr_to_file(file_name, rr_matrix, cols, rows, rr_prom):
    """Save the results of the investigations of R and m on the RR into a file.

    This should in the end print a beautifull csv file but for the moment it
    just saves everything
    """
    output = open(file_name, 'w')
    rr_template = "{:.3f},"*(len(rr_matrix[0]) - 1) + "{:.3f}"
    output.write("RR caused with Promethee: " + str(rr_prom) + "\n")
    output.write("values of m: " + str(cols) + "\n")
    output.write("values of R: " + str(rows) + "\n")
    output.write("rank reversals : \n")
    for row in rr_matrix:
        output.write(rr_template.format(*row))
        output.write("\n")
