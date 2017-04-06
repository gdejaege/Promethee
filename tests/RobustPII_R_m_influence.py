"""Test the number of RR for different R and m and different data sets."""
import promethee as prom
import data_reader as dr
import random


def count_rr(data='HDI', max_rep=10, R_parameter=None, m_parameter=None):
    """Test the number of rank reversals."""
    # Parameter initialization, the interesting stuff is way lower
    if(data == 'HDI'):
        # Change these parameters if needed
        R_list = [500, 1000, 5000, 10000]
        m_list = [3, 5, 6, 7, 8, 10, 15]

        # Do not change these parameters ! They are not saved
        data_set = 'data/HDI/raw.csv'
        alts = dr.open_raw(data_set)[0]
        weights = [0.5, 0.5]
        ceils = [3, 3]
        seed = 0                # Not used, here to match the general signature

    elif(data == 'SHA'):
        # Change these parameters if needed
        R_list = [1000, 4000, 7000, 12000]
        m_list = [4, 6, 8, 9, 12]

        # Do not change these parameters ! They are not saved
        data_set = 'data/SHA/raw_20.csv'
        alts = dr.open_raw(data_set)[0]
        weights = [0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667]
        ceils = [17.100, 23.7750, 26.100, 27.3750, 17.9250, 13.5750]
        seed = 0                # Not used, here to match the general signature

    else:
        # Change these parameters if needed
        R_list = [500, 1000, 5000, 8000]
        m_list = [18]

        # Do not change these parameters ! They are not saved
        data_set = 'data/EPI/raw.csv'
        alts = dr.open_raw(data_set)[0]
        alts = alts[0:20]
        weights, ceils = None, None
        seed = 0

    output_dir = 'res/RobustPII_R_m_influence/'
    output = output_dir + data + '.txt'

    promethee = prom.PrometheeII(alts, weights=weights, ceils=ceils, seed=seed)
    rr_promethee = promethee.compute_rr_number()

    rr_matrix = []
    for R in R_list:
        rr_row = []
        for m in m_list:
            rr = 0
            for repetition in range(max_rep):
                random.seed()
                robust = prom.RobustPII(alts, weights=weights, ceils=ceils,
                                        seed=seed, R=R, m=m)
                rr += robust.compute_rr_number()
            rr = rr/max_rep
            rr_row.append(rr)
        print(rr_row)
        rr_matrix.append(rr_row)
    print_rr_to_file(output, rr_matrix, R_list, m_list, rr_promethee, max_rep)


def test1(max_rep=20):
    """Test done for the 20 first alternatives of the HDI data set.

    Input:
        max_rep=number of repetitions of the method
    """
    output = 'res/R-m_in_RobustPII/test1.txt'

    # Change these parameters if needed
    R_list = [500, 1000, 5000, 10000]
    m_list = [3, 5, 6, 7, 8, 10, 15]

    # Do not change these parameters ! They are not saved
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
    print_rr_to_file(output, rr_matrix, R_list, m_list, rr_promethee, max_rep)


def test2(max_rep=10):
    """Test done for the 20 random alternatives of the SHA data set.

    Input:
        max_rep=number of repetitions of the method
    """
    output = 'res/R-m_in_RobustPII/test2.txt'

    # Change these parameters if needed
    R_list = [1000, 4000, 7000, 12000]
    m_list = [4, 6, 8, 9, 12]

    # Do not change these parameters ! They are not saved
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
    print_rr_to_file(output, rr_matrix, R_list, m_list, rr_promethee, max_rep)


def test3(max_rep=10):
    """Test done for the 20 first alternatives of the EPI data set.

    Input:
        max_rep=number of repetitions of the method
    """
    output = 'res/R-m_in_RobustPII/test3.txt'

    # Change these parameters if needed
    R_list = [500, 1000, 5000, 8000]
    m_list = [14, 16]

    # Do not change these parameters ! They are not saved
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
    print_rr_to_file(output, rr_matrix, R_list, m_list, rr_promethee, max_rep)


def print_rr_to_file(file_name, rr_matrix, rows, cols, rr_prom, max_rep):
    """Save the results of the investigations of R and m on the RR into a file.

    This should in the end print a beautifull csv file but for the moment it
    just saves everything
    """
    output = open(file_name, 'a')
    rr_template = "{:.3f},"*(len(rr_matrix[0]) - 1) + "{:.3f}"
    output.write("RR caused with Promethee: " + str(rr_prom) + "\n")
    output.write("values of m: " + str(cols) + "\n")
    output.write("values of R: " + str(rows) + "\n")
    output.write("average of: " + str(max_rep) + " repetitions \n")
    output.write("rank reversals : \n")
    for row in rr_matrix:
        output.write(rr_template.format(*row))
        output.write("\n")
    output.write("\n"*2)
