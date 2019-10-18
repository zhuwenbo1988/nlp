# coding=utf-8

import numpy as np
from numpy import *
import math


def viterbi(A, B, PI, O):
    N = shape(A)[0]
    T = len(O)
    I = np.zeros(T)
    sigma = mat(zeros((T, N)))
    omiga = mat(ones((T, N)))

    t = 0
    o_0 = O[t]
    for i in range(N):
        sigma[t, i] = PI[i] * B[i, o_0]
    
    t = 1
    while(t < T):
        for i in range(N):
            sigma_temp = np.zeros(N)
            for j in range(N):
                sigma_temp[j] = sigma[t - 1, j] * A[j, i]
            max_value = sigma_temp.max()
            o_t = O[t]
            sigma[t, i] = max_value * B[i, o_t]
            # N: 1->N
            omiga[t, i] = sigma_temp.argmax() + 1
        t += 1
    P = sigma[T - 1, :].max()
    I[T - 1] = sigma[T - 1, :].argmax() + 1
    t = T - 2

    while(t >= 0):
        last_idx = int(I[t + 1] - 1)
        I[t] = omiga[t + 1, last_idx]
        t -= 1
    return I


if __name__ == "__main__":
    A = mat([[0.5, 0.2, 0.3],
             [0.3, 0.5, 0.2],
             [0.2, 0.3, 0.5]])
    B = mat([[0.5, 0.5],
             [0.4, 0.6],
             [0.7, 0.3]])
    PI = [0.2, 0.4, 0.4]
    O = [0, 1, 0]
    I = viterbi(A, B, PI, O)
    print(I)
