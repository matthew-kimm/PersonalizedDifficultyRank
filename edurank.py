from typing import List, Tuple
import numpy as np
from partial_order import partial_order, partial_order_on_items
from metrics import ap_score_how


def relative_votings(ap_scores: np.array, pos: np.array):
    rvs = np.sum(ap_scores[np.newaxis, np.newaxis, :] * pos, axis=2)
    return np.sign(rvs)


def copeland_score(rvs: np.array):
    return np.sum(rvs, axis=1, keepdims=True)


def edurank(ap_scores: np.array, test_pos: np.array):
    return partial_order(copeland_score(relative_votings(ap_scores, test_pos)))


def ap_scores_neighbors(train_users: List[Tuple[np.array, dict, np.array]],
                        test_users: List[Tuple[np.array, dict, np.array]],
                        train_test_same: bool = False, how: str = ''):
    n = len(train_users)
    m = len(test_users)
    result = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            hpo_1, hmap_1, hite_1, hn_1, _, _, _, _ = test_users[i]
            hpo_2, hmap_2, _, hn_2 = train_users[j]
            result[i, j] = ap_score_how(partial_order_on_items(hpo_1, hite_1, hmap_1, hn_1),
                                        partial_order_on_items(hpo_2, hite_1, hmap_2, hn_2),
                                        how=how)
    if train_test_same:
        np.fill_diagonal(result, 0)
    return np.nan_to_num(result, 0)


def edurank_compute(train_users: list, test_users: list, train_test_same: bool = False, how: str = '',
                    use_ap: bool = True):
    if use_ap:
        ap_scores = ap_scores_neighbors(train_users, test_users, train_test_same=train_test_same, how=how)
    else:
        ap_scores = np.ones((len(test_users), len(train_users)))
        if train_test_same:
            np.fill_diagonal(ap_scores, 0)
    proposed_pos = []
    for i, tu in enumerate(test_users):
        _, _, _, _, _, _, tite_1, _ = tu
        test_pos = np.dstack([partial_order_on_items(tru[0], tite_1, tru[1], tru[3]) for tru in train_users])
        proposed_po = edurank(ap_scores[i, :], test_pos)
        proposed_pos.append(proposed_po)
    return proposed_pos
