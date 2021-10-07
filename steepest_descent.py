#
# JSOPT: python project for optimization theory class, 2021 fall
#
# steepest_descent.py: python package for steepest descent algorithm
#
# Developed and Maintained by Soonkyu Jeong (reingel@o.cnu.ac.kr)
#  since Oct. 1, 2021
#


import numpy as np
from constant import *
from linalg import Vector, normalized
from line_search import bisection


def sda(f, gradf, x0, epsilon=1e-6, max_num_iter=1000, line_search=bisection, alpha_max=99, ls_epsilon=1e-6, ls_max_num_iter=1000):
	"""
	minimize f(x) using the steepest descent algorithm
	sda() function description
	1. input arguments
		- f: an objective function f(x) (function)
		- gradf: the gradient of f(x) (function)
		- x0: a starting point of optimization (Vector = numpy.ndarray)
		- epsilon: the first stopping criteria. sda() will stop if |gradf(xk)| <= epsilon. (float)
		- max_num_iter: the second stopping criteria. sda() will stop if the number of iterations is greater than max_num_iter. (integer)
		- line_search: the function of line search algorithm
		- alpha_max: the max. range of line search (numeric of a function of x and d)
		- ls_epsilon: epsilon for line search algorithm
		- ls_max_num_iter: max_num_iter for line search algorithm
	2. return values
		- x_opt: the minimizer of f(x) (Vector = numpy.ndarray)
		- fval_opt: the minimum of f(x) (float)
		- status: 0 if the minimum is found within max_num_iter, 1 if the number of iterations reaches max_num_iter. (integer)
		- history: sequencially stored values of x, d, fval, rate_conv (dictionary)
	"""

	# set initial values
	k = 0
	xk = x0
	fk = f(xk)
	dk = -gradf(xk)

	# define a line search function g'(a)=gradf(x0 + a * d).d ('.' means dot product)
	xd = lambda alpha: x0 + alpha * dk
	gradfd = lambda alpha: np.dot(gradf(xd(alpha)), dk)

	# create additional return values
	history = {'x': [xk], 'd': [dk], 'fval': [fk]}
	status = CONVERGED

	# search loop
	while (np.linalg.norm(dk) > epsilon): # stopping criteria 1
		alpha_max_value = alpha_max if np.isscalar(alpha_max) else alpha_max(xk, dk)
		alpha_k, _, _, _ = line_search(gradfd, alpha_max=alpha_max_value, epsilon=ls_epsilon, max_num_iter=ls_max_num_iter)
		xk = xd(alpha_k)
		fk = f(xk)
		dk = -gradf(xk)

		# store histories
		history['x'].append(xk)
		history['d'].append(dk)
		history['fval'].append(fk)

		k += 1
		if k == max_num_iter: # stopping criteria 2
			status = REACHED_MAX_ITER
			print(f'sda: reached the maximum number of iteration: {k}')
			break
	
	x_opt = xk
	fval_opt = fk

	# convert to numpy array
	history['x'] = np.array(history['x'])
	history['d'] = np.array(history['d'])
	history['fval'] = np.array(history['fval'])

	# add rate of convergence to history
	history['rate_conv'] = (history['fval'][1:] - fval_opt) / (history['fval'][:-1] - fval_opt)
	history['rate_conv'] = np.insert(history['rate_conv'], 0, 1.)

	return x_opt, fval_opt, status, history


# test code
if __name__ == '__main__':
	import matplotlib.pyplot as plt
	
	f = lambda x: x[0]**2 + 2*(x[1]-1)**2
	gradf = lambda x: Vector([2*x[0], 4*(x[1]-1)])
	x0 = Vector([-2, 1.4])

	def calc_alpha_max(x, d):
		alpha_max = 2
		if d[1] > 0:
			alpha_max = min(alpha_max, (0.5 - x[1]) / d[1])
		return alpha_max

	x_opt, fval_opt, status, history = sda(f, gradf, x0, epsilon=1e-3, max_num_iter=100, line_search=bisection, alpha_max=calc_alpha_max)

	print(f"sda: {status=}, x_opt={np.round(x_opt,2)}, fval_opt={np.round(fval_opt,2)}, num_iter={len(history['x'])}")

	fig, ax = plt.subplots(2,1)
	ax[0].set_aspect(1.0)
	ax[0].grid(True)
	x = np.linspace(-3,3,50)
	y = np.linspace(-3,2,40)
	[xx, yy] = np.meshgrid(x, y)
	zz = np.zeros_like(xx)
	for i in range(len(x)):
		for j in range(len(y)):
			zz[j,i] = f((x[i],y[j]))
	ax[0].contour(xx, yy, zz)
	ax[0].scatter(history['x'][:,0], history['x'][:,1])
	ax[0].scatter(history['x'][0,0], history['x'][0,1], marker='^', color='green')
	ax[0].scatter(history['x'][-1,0], history['x'][-1,1], color='red')
	ax[1].grid(True)
	# ax[1].plot(history['d'], label='d')
	ax[1].plot(history['fval'], label='fval')
	ax[1].plot(history['rate_conv'], label='rate_conv')
	ax[1].legend(loc='best')
	plt.show()
