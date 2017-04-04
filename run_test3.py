"""This file is used to launch the different test files."""
import time
from tests import *

t1 = time.time()

# test1_datareader.test()

# tests_class_promethee.test1()
# tests_class_promethee.test2()
# tests_class_promethee.test3()

tests_R_m_in_RobustPII.test3()

t2 = time.time()
print('test durations ::' + str(t2-t1))
