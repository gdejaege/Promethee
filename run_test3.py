"""This file is used to launch the different test files."""
import time
from tests import *

t1 = time.time()

# test1_datareader.test()

# tests_class_promethee.test1()
# tests_class_promethee.test2()
# tests_class_promethee.test3()

# tests_class_RobustPII.test2(20)
tests_class_RobustPII.test3(10)
# tests_class_RobustPII.test4(1)

t2 = time.time()
print('test durations ::' + str(t2-t1))
