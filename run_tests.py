"""This file is used to launch the different test files."""
import time
from tests import *

t1 = time.time()

# tests_data_reader.test1()

data = ['HDI', 'SHA', 'EPI']
# Promethee_class.test_ranking()
# Promethee_class.test_rr_counting_function()
# Promethee_class.test_rr_analysis(data[1])

# RobustPII_class.test_ranking()

# RobustPII_R_m_influence.count_rr(data=data[2], max_rep=10)

RobustPII_analyse_rank_reversals.analyse_rr(data=data[2], m_parameter=16,
                                            R_parameter=5000, max_rep=10)

# ReferencedPII_references_quantity.count_draws(threshold=1e-5)

t2 = time.time()
print('test durations ::' + str(t2-t1))
