#
# JSOPT: python project for optimization theory class, 2021 fall
#
# LinAlg.py: python package for linear algebra
#
# Developed and Maintained by Soonkyu Jeong (reingel@o.cnu.ac.kr)
#  since Oct. 1, 2021
#


import numpy as np


Vector = np.array
Matrix = np.array

def normalized(vector):
	return vector / np.linalg.norm(vector)
