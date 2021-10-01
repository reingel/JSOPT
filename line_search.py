#
# line_search.py
# The line search library for optimization
# JSOPT: Programming Homeworks of Optimization Theory Class, 2021 Fall
#
# Developed and Maintained by Soonkyu Jeong (reingel@gmail.com)
# Since Oct. 1, 2021
#


import numpy as np

Vector = np.array

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
        k += 1
        if k > max_loop:
            print(f'Reached the maximum number of iteration: {max_loop}')
            break
        alpha_tilde = (alpha_l + alpha_u) / 2
        grad = dg(alpha_tilde)
    
    return alpha_tilde, k


if __name__ == '__main__':
    
    f = lambda x,y: x**2 + y**2
    gradf = lambda x,y: Vector((2*x, 2*y))

    xk = Vector((-1, 1))
    d = Vector((1, 0))
    max_step = 4

    xd = lambda alpha: xk + alpha * d
    dg = lambda alpha: gradf(xd(alpha))

    alpha, k = bisection(dg, max_step)

    xk1 = xk + alpha * d

    print(f'{xk=}, {xk1=}, {alpha=}, {k=}')