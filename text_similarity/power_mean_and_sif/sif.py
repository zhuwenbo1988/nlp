# coding=utf-8

from sklearn.decomposition import TruncatedSVD

def compute_pc(X, npc=1):
    svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
    svd.fit(X)
    return svd.components_

def remove_pc(v, pc):
    vv = v - v.dot(pc.transpose()) * pc
    return vv

# 先计算主成分(compute_pc),再减去主成分(remove_pc)
