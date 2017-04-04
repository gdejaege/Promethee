"""This file is used to launch the different test files."""
import time
from tests import *

t1 = time.time()

# tests_data_reader.test1()

# tests_class_promethee.test1()
# tests_class_promethee.test2()
# tests_class_promethee.test3()
# tests_class_promethee.test4()
tests_class_promethee.test5()

# Robust PII
# tests_class_RobustPII.test2(20)

# tests_R_m_in_RobustPII.test1()
# tests_R-m_in_RobustPII.test2()
# tests_R-m_in_RobustPII.test3()

t2 = time.time()
print('test durations ::' + str(t2-t1))
