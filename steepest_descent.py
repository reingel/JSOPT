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


def sda(f, gradf, x0, epsilon=1e-6, max_num_iter=1000, line_search=bisection, ls_epsilon=1e-6, ls_max_num_iter=1000):
	# minimize f(x) using the steepest descent algorithm
	# sda() function description
	# 1. input arguments
	# 	- f: an objective function f(x) (function)
	# 	- gradf: the gradient of f(x) (function)
	# 	- x0: a starting point of optimization (Vector = numpy.ndarray)
	# 	- epsilon: the first stopping criteria. sda() will stop if |gradf(xk)| <= epsilon. (float)
	# 	- max_num_iter: the second stopping criteria. sda() will stop if the number of iterations is greater than max_num_iter. (integer)
	# 2. return values
	# 	- xopt: the minimizer of f(x) (Vector = numpy.ndarray)
	# 	- fval_opt: the minimum of f(x) (float)
	# 	- status: 0 if the minimum is found within max_num_iter, 1 if the number of iterations reaches max_num_iter. (integer)
	# 	- history: the sequencially stored values of x, d, fval (dictionary)
	k = 0
	xk = x0
	fk = f(xk)
	dk = -gradf(xk)

	history = {'x': [xk], 'd': [dk], 'fval': [fk]}
	status = CONVERGED

	while (np.linalg.norm(dk) > epsilon):
		xk, fk, _, _ = line_search(f, gradf, xk, dk, alpha_max=100, epsilon=ls_epsilon, max_num_iter=ls_max_num_iter)
		dk = -gradf(xk)

		history['x'].append(xk)
		history['d'].append(dk)
		history['fval'].append(fk)

		k += 1
		if k == max_num_iter:
			status = REACHED_MAX_ITER
			print(f'sda: reached the maximum number of iteration: {k}')
			break
	
	xopt = xk
	fval_opt = fk

	history['x'] = np.array(history['x'])
	history['d'] = np.array(history['d'])
	history['fval'] = np.array(history['fval'])
	history['rate_conv'] = (history['fval'][1:] - fval_opt) / (history['fval'][:-1] - fval_opt)
	history['rate_conv'] = np.insert(history['rate_conv'], 0, 1.)

	return xopt, fval_opt, status, history


if __name__ == '__main__':
	import matplotlib.pyplot as plt
	
	f = lambda x: x[0]**2 + 2*(x[1]-1)**2
	gradf = lambda x: Vector([2*x[0], 4*(x[1]-1)])
	x0 = Vector([-2, 1.4])

	xopt, fval_opt, status, history = sda(f, gradf, x0, epsilon=1e-3, max_num_iter=1000, line_search=bisection)

	print(f"sda: {status=}, xopt={np.round(xopt,2)}, fval_opt={np.round(fval_opt,2)}, num_iter={len(history['x'])}")

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
