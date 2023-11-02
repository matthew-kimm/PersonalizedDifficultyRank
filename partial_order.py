import numpy as np


def partial_order(x: np.array):
    n = len(x)
    row_repeat = np.repeat(x, n, axis=1)
    col_repeat = row_repeat.T
    po = np.sign(row_repeat - col_repeat)
    padded_po = np.pad(po, pad_width=((0, 1), (0, 1)), mode="constant")
    return padded_po


def partial_order_on_items(po: np.array, items: np.array, idx_map: np.array, default: int):
    idxs = [idx_map.get(item, default) for item in items]
    np_idxs = np.ix_(idxs, idxs)
    return po[np_idxs]


def votes_on_items(po: np.array, items_x: np.array, items_y, idx_map: np.array, default: int):
    idxs_x = [idx_map.get(item, default) for item in items_x]
    idxs_y = [idx_map.get(item, default) for item in items_y]
    np_idxs = np.ix_(idxs_x, idxs_y)
    return po[np_idxs]
