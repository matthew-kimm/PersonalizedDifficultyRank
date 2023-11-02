import numpy as np


def ap_score_how(po_1: np.array, po_2: np.array, how: str = ''):
    if not how:
        return ap_score(po_1, po_2)
    elif how == 'reverse':
        return ap_score(po_2, po_1)
    elif how == 'symmetric':
        return (ap_score(po_1, po_2) + ap_score(po_2, po_1)) / 2
    else:
        raise NotImplementedError('Only variations are reverse and symmetric')


def ap_score(po_1: np.array, po_2: np.array):
    more_difficult = np.nonzero(po_1 < 0)
    more_difficult_x = more_difficult[0]
    m = len(more_difficult_x)
    if not m:
        return np.nan
    elif m == 1:
        return int(po_1[more_difficult] == po_2[more_difficult])

    idx = np.where(more_difficult_x[1:] != more_difficult_x[:-1])[0] + 1
    vals = np.diff(np.concatenate([[0], idx, [m]]))
    div = np.repeat(vals, vals)
    points = (po_1[more_difficult] == po_2[more_difficult]) / div
    result = np.sum(points) / len(vals)
    return result


def ndpm_score_how(po_1: np.array, po_2: np.array, how: str = ''):
    if not how:
        return ndpm_score(po_1, po_2)
    elif how == 'reverse':
        return ndpm_score(po_2, po_1)
    elif how == 'symmetric':
        return (ndpm_score(po_1, po_2) + ndpm_score(po_2, po_1)) / 2


def ndpm_score(po_1: np.array, po_2: np.array):
    ref = np.triu(po_1)
    prop = np.triu(po_2)
    nz = np.where(ref != 0)
    n = len(nz[0])
    if not n:
        return np.nan
    incorrect = np.sum((ref[nz] != prop[nz]) * (prop[nz] != 0))
    compatible = np.sum((ref[nz] != 0) * (prop[nz] == 0))
    ndpm = (2 * incorrect + compatible) / (2 * n)
    return ndpm
