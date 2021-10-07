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


def bisection(gradfd, alpha_max, epsilon=1e-6, max_num_iter=1000):
	"""
	minimize f(x0 + alpha * d) w.r.t. alpha using the bisection algorithm
	bisection() function description
	1. input arguments
		- gradfd: the gradient of f(x + alpha*d) (function)
		- alpha_max: the maximum search length of alpha
		- epsilon: the first stopping criteria. bisection() will stop if |gradf(xk).d| <= epsilon. (float)
		- max_num_iter: the second stopping criteria. bisection() will stop if the number of iterations is greater than max_num_iter. (integer)
	2. return values
		- xopt: the minimum point of f(x0 + alpha * d) (Vector = numpy.ndarray)
		- fval_opt: the minimum of f(x0 + alpha * d) (float)
		- status: 0 if the minimum is found within max_num_iter, 1 if the number of iterations reaches max_num_iter. (integer)
		- history: sequencially stored values of alpha, x, fval (dictionary)
	"""

	# set initial values
	k = 0
	alpha_l, alpha_u = 0., alpha_max
	alpha_k = (alpha_l + alpha_u) / 2
	grad0 = gradfd(alpha_l)
	grad_k = gradfd(alpha_k)

	# create additional return values
	history = {'alpha': [alpha_l, alpha_k], 'grad': [grad0, grad_k]}
	status = CONVERGED

	# search loop
	while (abs(grad_k) > epsilon): # stopping criteria 1
		if grad_k > epsilon:
			alpha_u = alpha_k
		elif grad_k < -epsilon:
			alpha_l = alpha_k
		alpha_k = (alpha_l + alpha_u) / 2
		grad_k = gradfd(alpha_k)

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
	# x = xd(alpha)
	# fval = f(x)

	print(f"bisection: {status=}, alpha_opt={np.round(alpha_opt,2)}, x_opt={np.round(x_opt,2)}, grad_opt={np.round(grad_opt,2)}, num_iter={len(alpha)}")

	# fig, ax = plt.subplots(2,1)
	# ax[0].set_aspect(1.0)
	# ax[0].grid(True)
	# x = np.linspace(-3,3,50)
	# y = np.linspace(-3,3,50)
	# [xx, yy] = np.meshgrid(x, y)
	# zz = np.zeros_like(xx)
	# for i in range(len(x)):
	# 	for j in range(len(y)):
	# 		zz[i,j] = f((x[i],y[j]))
	# ax[0].contour(xx, yy, zz)
	# ax[0].scatter(x[:,0], x[:,1])
	# ax[0].scatter(x[0,0], x[0,1], color='green')
	# ax[0].scatter(x[-1,0], x[-1,1], color='red')
	# ax[1].grid(True)
	# ax[1].plot(alpha, label='alpha')
	# ax[1].plot(fval, label='fval')
	# ax[1].legend(loc='best')
	# plt.show()
