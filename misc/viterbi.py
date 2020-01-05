# coding=utf-8

import numpy as np
from numpy import *
import math


def viterbi(A, B, PI, X):
    K = shape(A)[0]
    T = len(X)
    Y = np.zeros(T)
    path_score = mat(zeros((T, K)))
    path_index = mat(ones((T, K)))

    o_0 = X[0]
    for i in range(K):
        path_score[0, i] = PI[i] * B[i, o_0]
    
    t = 1
    while(t < T):
        for i in range(K):
            j_to_i_score = np.zeros(K)
            for j in range(K):
                j_to_i_score[j] = path_score[t - 1, j] * A[j, i]
            o_t = X[t]
            path_score[t, i] = j_to_i_score.max() * B[i, o_t]
            # K: 1,2,3,...,K
            path_index[t, i] = j_to_i_score.argmax() + 1
        t += 1
    Y[T - 1] = path_score[T - 1, :].argmax() + 1
    t = T - 2

    while(t >= 0):
        last_hidden_state = int(Y[t + 1] - 1)
        Y[t] = path_index[t + 1, last_hidden_state]
        t -= 1
    return Y


if __name__ == "__main__":
    # K*K
    A = mat([[0.5, 0.2, 0.3],
             [0.3, 0.5, 0.2],
             [0.2, 0.3, 0.5]])
    # K*N
    B = mat([[0.5, 0.5],
             [0.4, 0.6],
             [0.7, 0.3]])
    # K
    PI = [0.2, 0.4, 0.4]
    # T
    X = [0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1]
    Y = viterbi(A, B, PI, X)
    print(Y)
