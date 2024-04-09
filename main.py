from scipy.optimize import fsolve
from sympy import Symbol, nsolve
import numpy as np
import math


a = np.array([[  1 /   4,            0,         0,        0,     0],
              [  1 /   2,     1 /    4,         0,        0,     0],
              [ 17 /   50,   -1 /   25,   1 /   4,        0,     0],
              [371 / 1360, -137 / 2720,  15 / 544,   1 /  4,     0],
              [ 25 /   24,  -49 /   48, 125 /  16, -85 / 12, 1 / 4]])

b = np.array([  25 /   24, - 49 /   48, 125 /  16, -85 / 12, 1 / 4])


def compute_ks(fs, y, step, n):
    s = len(fs)

    # create all variables
    ks = [[Symbol(f"k_{i}{j}") for j in range(s)] for i in range(n)]

    equations = []
    for i in range(n): # TODO: error, should be dependent on a's size
        for j in range(s):
            full_k = np.array([ks[i][m] for m in range(s)])

            sum_set = [a[i, m] * full_k for m in range(j + 1)]
            vector_argument = y + step * sum(sum_set)

            equation = fs[j](*vector_argument) - ks[i][j]
            equations.append(equation)

    print(equations)
    answers = nsolve(equations, ks, tuple([1] * n))

    vectorized_answers = []
    for i in range(n, step=s):
        # pack answer in np.array
        answer = np.array([answers[j] for j in range(i, i + s)])
        vectorized_answers.append(answer)

    return vectorized_answers

def compute_ys(fs, y0, step, n_max, n):
    ys = [y0] * n_max

    for i in range(n_max - 1):
        ks = compute_ks(fs, ys[i], step, n)
        ys[i + 1] = ys[i] + step * sum([ b[j] * ks[j] for j in range(n) ])

    return ys


def main():
    fs = [
        lambda y1, y2, y3: 77.27 * (y2 + y1 * (1 - 8.375 * math.pow(10, -6) * y1 - y2)),
        lambda y1, y2, y3: (1 / 77.27) * (y3 - (1 + y1) * y2),
        lambda y1, y2, y3: 0.16 * (y1 - y3)
    ]

    result = compute_ys(fs, np.array([1, 1, 1]), 0.01, 1000, 3)
    print(result)


if __name__ == '__main__':
    main()
