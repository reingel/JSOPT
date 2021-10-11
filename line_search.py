#
# JSOPT: python project for optimization theory class, 2021 fall
#
# line_search.py: python package for line search algorithm
#
# Developed and Maintained by Soonkyu Jeong (reingel@o.cnu.ac.kr)
#  since Oct. 1, 2021
#


import numpy as np
from constant import *
from linalg import Vector, normalized


def bisection(gradf, alpha_max, epsilon=1e-6, max_num_iter=1000):
	"""
	bisection(): find the minimimum point of a 1D function using the bisection algorithm
	<function description>
	1. input arguments
		- gradf: the gradient of f(alpha) (function)
		- alpha_max: the maximum search length of alpha
		- epsilon: the 1st stopping criteria. bisection() will stop if |gradf(alpha)| <= epsilon. (float)
		- max_num_iter: the 2nd stopping criteria. bisection() will stop if the number of iterations is greater than max_num_iter. (integer)
	2. return values
		- alpha_opt: the minimum point of f(alpha) (float)
		- grad_opt: the gradient of f(alpha) at alpha_opt (float)
		- status: (integer)
			- 0 if the minimum point is found within max_num_iter
			- 1 if the number of iterations reaches max_num_iter
		- history: sequencially stored values of alpha, grad (dictionary)
	"""

	# set initial values
	k = 0
	alpha_l, alpha_u = 0., alpha_max
	alpha_k = alpha_l
	grad_k = gradf(alpha_l)

	# create additional return values
	status = CONVERGED
	history = {'alpha': [alpha_k], 'grad': [grad_k]}

	# search loop
	while (abs(grad_k) > epsilon): # stopping criteria 1 (|gradf(alpha) == 0|)
		if grad_k > epsilon: # gradf(alpha) > 0
			alpha_u = alpha_k
		elif grad_k < -epsilon: # gradf(alpha) < 0
			alpha_l = alpha_k
		alpha_k = (alpha_l + alpha_u) / 2
		grad_k = gradf(alpha_k) # update the gradient

		# store histories
		history['alpha'].append(alpha_k)
		history['grad'].append(grad_k)

		k += 1
		if k == max_num_iter: # stopping criteria 2
			status = REACHED_MAX_ITER
			print(f'bisection: reached the maximum number of iteration: {k}')
			break

	# solutions to return
	alpha_opt = alpha_k
	grad_opt = grad_k

	# convert to numpy array
	history['alpha'] = np.array(history['alpha'])
	history['grad'] = np.array(history['grad'])

	return alpha_opt, grad_opt, status, history


# test code
if __name__ == '__main__':
	import matplotlib.pyplot as plt
	
	f = lambda x: x[0]**2 + x[1]**2
	gradf = lambda x: Vector([2*x[0], 2*x[1]])
	
	x0 = Vector([-2, 1])
	d = normalized(Vector([1, -1]))

	# define a line search function g'(a)=gradf(x0 + a * d).d ('.' means dot product)
	xd = lambda alpha: x0 + alpha * d
	gradfd = lambda alpha: np.dot(gradf(xd(alpha)), d)

	alpha_max = 4.6

	alpha_opt, grad_opt, status, history = bisection(gradfd, alpha_max)
	x_opt = xd(alpha_opt)
	alpha = history['alpha']

	print(f"bisection: {status=}, alpha_opt={np.round(alpha_opt,2)}, x_opt={np.round(x_opt,2)}, grad_opt={np.round(grad_opt,2)}, num_iter={len(alpha)-1}")
