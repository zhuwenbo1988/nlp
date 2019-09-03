# coding=utf-8

import numpy as np

def power_mean(v1, v2):
    v1_mean = np.mean(v1, axis=0)
    v2_mean = np.mean(v2, axis=0)
    v1_max = np.amax(v1, axis=0)
    try:
        v2_max = np.amax(v2, axis=0)
    except Exception as e:
        v2_max = np.zeros(v1_max.shape)
    v1_min = np.amin(v1, axis=0)
    try:
        v2_min = np.amin(v2, axis=0)
    except Exception as e:
        v2_min = np.zeros(v1_min.shape)
    v1 = np.concatenate([v1_mean, v1_max, v1_min])
    v2 = np.concatenate([v2_mean, v2_max, v2_min])
    return v1, v2
