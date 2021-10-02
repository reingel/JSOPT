#
# JSOPT: python project for optimization theory class, 2021 fall
#
# steepest_descent.py: python package for steepest descent algorithm
#
# Developed and Maintained by Soonkyu Jeong (reingel@o.cnu.ac.kr)
#  since Oct. 1, 2021
#


import numpy as np
from Vector import Vector, normalized
from line_search import bisection


def sda(gradf, x0, epsilon=1e-6, max_loop=1000, line_search=bisection, ls_epsilon=1e-6, ls_max_loop=1000):
    # TODO: store every k, xk, dk, ...
    k = 0
    xk = x0
    dk = -gradf(*xk)

    while (np.linalg.norm(dk) > epsilon):
        xd = lambda alpha: xk + alpha * dk
        dg = lambda alpha: np.dot(gradf(*xd(alpha)), dk)
        alpha, _ = line_search(dg, max_step=100, epsilon=ls_epsilon, max_loop=ls_max_loop) # TODO: deal k
        xk = xk + alpha * dk
        k += 1
        if k == max_loop:
            print(f'sda(): reached the maximum number of iteration: {k}')
            break
        dk = -gradf(*xk)
    
    return xk, k


if __name__ == '__main__':
    
    f = lambda x,y: x**2 + 2*(y-1)**2
    gradf = lambda x,y: Vector([2*x, 4*(y-1)])

    x0 = Vector([-2, 1.4])
    xopt, k = sda(gradf, x0)

    print(f'{x0=}, {xopt=}, {k=}')