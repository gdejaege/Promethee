"""This file is used to launch the different test files.

Be aware that some tests could take hours (at least on my laptop).
All the parameters of the tests are optional. However, its should be easier for
you to define them.
"""
import time
from tests import *

t1 = time.time()
data = ['HDI', 'SHA', 'EPI', 'GEQ']

# Some tests to check whether the functions are perfoming correctly. #
######################################################################

# data_reader.test()
# Promethee_class.test_ranking()
# Promethee_class.test_rr_counting_function()
# Promethee_class.test_rr_analysis(data[1])
# RobustPII_class.test_ranking()
# ReferencedPII_class.compare_refflows()
# ReferencedPII_questioning_procedure.test_functions()

# Test performed to build the tables in section 4.2 of the master thesis. #
###########################################################################
# Tables 4.1, 4.2, 4.4
# Set R and m to None to perform the same tests as in the master thesis. Be
# however aware that it should take quite some time.
R = [500, 1000, 10000]
m = [4, 5, 7, 8]
# RobustPII_R_m_influence.count_rr(data=data[0], max_rep=2, R_parameter=R,
#                                  m_parameter=m)

# Tables 4.3, 4.5
R = 5000
m = 16
max_rep = 10
# RobustPII_analyse_rank_reversals.analyse_rr(data=data[2], m_parameter=m,
#                                             R_parameter=R, max_rep=max_rep)


# Test performed to build the table in section 5.1.1 of the master thesis. #
############################################################################
# ReferencedPII_references_quantity.count_draws(threshold=1e-3)


# Test performed to build the tables in section 5.1.2 of the master thesis. #
#############################################################################
# reminder: data = ['HDI', 'SHA', 'EPI', 'GEQ']
pop_size = 600
mut_prob = 0.01
MAXIT = 5
# ReferencedPII_genetic_search.first_search(pop_size=pop_size,
#                                           mut_prob=mut_prob,
#                                           MAXIT=MAXIT)
# failed seeds must be composed of one list of seeds for each alternative
# quantity ! ex :
#     alternatives_qty = [20, 30]
#     failed_seeds = [[4], [0, 6]]
alternatives_qty = [20]
failed_seeds = [[4]]
# ReferencedPII_genetic_search.retry_failed(data[2], alternatives_qty,
#                                           failed_seeds, pop_size=pop_size,
#                                           mut_prob=mut_prob, MAXIT=MAXIT)


# Test performed to build the tables in section 5.2.1 of the master thesis. #
#############################################################################
quantity_of_samples = 1000
# ReferencedPII_strategies_comparison.compare(quantity_of_samples)

# Test performed to build the tables in section 5.2.2 of the master thesis. #
#############################################################################
# * This test does not print to a file by default, to print to a file please
#   uncomment line 'with redirect...' (and comment the line above of the file
#   ReferencedPII_questioning_procedure)
# * These test could take quite some time (several hours) with larger sets of
#   alternatives or more rounds
seeds = range(1)
alt_num = 15
data_set = [data[1]]   # must be a list
rounds = 10
points_per_it = 150
max_random_add_it = 100
make_pdf = False
# ReferencedPII_questioning_procedure.analyse(alt_num=alt_num, seeds=seeds,
#                                             data_sets=data_set, rounds=rounds,
#                                             make_pdf=make_pdf)


# Test performed to build the tables in section 5.2.2 of the master thesis. #
#############################################################################
data_set = data[3]
# ReferencedPII_search_SRP_properties.SRP_from_aqp(data_set=data_set)

# The parallel coordinate box plot has been realised with an R script. Please
# test it directly in the 'tests' folder by lanching:
# $ Rscript ReferencedPII_plot_parallel_coordinates.R

t2 = time.time()
print('test durations ::' + str(t2-t1))
