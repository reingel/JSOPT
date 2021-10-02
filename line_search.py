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


def bisection(f, gradf, x0, d, alpha_max, epsilon=1e-6, max_num_iter=1000):
    xd = lambda alpha: x0 + alpha * d
    dg = lambda alpha: np.dot(gradf(xd(alpha)), d)

    k = 0
    alpha_l, alpha_u = 0., alpha_max
    alpha = (alpha_l + alpha_u) / 2
    x = xd(alpha)
    fval = f(x)
    grad = dg(alpha)

    history = {'alpha': [alpha_l,alpha], 'x': [x0,x], 'fval': [f(x0),fval]}
    status = CONVERGED

    while (abs(grad) > epsilon):
        if grad > epsilon:
            alpha_u = alpha
        elif grad < -epsilon:
            alpha_l = alpha
        alpha = (alpha_l + alpha_u) / 2
        x = xd(alpha)
        fval = f(x)
        grad = dg(alpha)

        history['alpha'].append(alpha)
        history['x'].append(x)
        history['fval'].append(fval)

        k += 1
        if k == max_num_iter:
            status = REACHED_MAX_ITER
            print(f'bisection(): reached the maximum number of iteration: {k}')
            break

    history['alpha'] = np.array(history['alpha'])
    history['x'] = np.array(history['x'])
    history['fval'] = np.array(history['fval'])

    xopt = x
    fval_opt = fval

    return xopt, fval_opt, status, history


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    f = lambda x: x[0]**2 + x[1]**2
    gradf = lambda x: Vector([2*x[0], 2*x[1]])
    x0 = Vector([-2, 1])
    d = normalized(Vector([1, -1]))
    alpha_max = 4.6

    xopt, fval_opt, status, history = bisection(f, gradf, x0, d, alpha_max)

    print(f'xopt={np.round(xopt,2)}, fval_opt={np.round(fval_opt,2)}')

    fig, ax = plt.subplots(2,1)
    ax[0].set_aspect(1.0)
    ax[0].grid(True)
    x = np.linspace(-3,3,50)
    y = np.linspace(-3,3,50)
    [xx, yy] = np.meshgrid(x, y)
    zz = np.zeros_like(xx)
    for i in range(len(x)):
        for j in range(len(y)):
            zz[i,j] = f((x[i],y[j]))
    ax[0].contour(xx, yy, zz)
    ax[0].scatter(history['x'][:,0], history['x'][:,1])
    ax[0].scatter(history['x'][0,0], history['x'][0,1], color='green')
    ax[0].scatter(history['x'][-1,0], history['x'][-1,1], color='red')
    ax[1].grid(True)
    ax[1].plot(history['alpha'], label='alpha')
    ax[1].plot(history['fval'], label='fval')
    ax[1].legend(loc='best')
    plt.show()
