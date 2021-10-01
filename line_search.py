#
# JSOPT: python project for optimization theory class, 2021 fall
#
# line_search.py: python package for line search algorithm
#
# Developed and Maintained by Soonkyu Jeong (reingel@o.cnu.ac.kr)
#  since Oct. 1, 2021
#


import numpy as np
from Vector import *


def bisection(dg, max_step, epsilon=1e-3, max_loop=1000):
    k = 0
    alpha_l, alpha_u = 0., max_step

    alpha_tilde = (alpha_l + alpha_u) / 2
    grad = dg(alpha_tilde)

    while (abs(grad) > epsilon):
        if grad > epsilon:
            alpha_u = alpha_tilde
        elif grad < -epsilon:
            alpha_l = alpha_tilde
        alpha_tilde = (alpha_l + alpha_u) / 2
        grad = dg(alpha_tilde)
        k += 1
        if k == max_loop:
            print(f'bisection(): reached the maximum number of iteration: {max_loop}')
            break
    
    return alpha_tilde, k


if __name__ == '__main__':
    
    f = lambda x,y: x**2 + y**2
    gradf = lambda x,y: Vector([2*x, 2*y])

    xk = Vector([-2, 1])
    d = normed(Vector([1, -1]))
    max_step = 4.6

    xd = lambda alpha: xk + alpha * d
    dg = lambda alpha: np.dot(gradf(*xd(alpha)), d)

    alpha, k = bisection(dg, max_step)

    xk1 = xk + alpha * d

    print(f'{xk=}, {xk1=}, {alpha=}, {k=}')