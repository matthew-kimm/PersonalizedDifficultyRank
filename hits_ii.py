import numpy as np
from partial_order import partial_order, votes_on_items
from edurank import ap_scores_neighbors


def hits_power_iteration(matrix: np.array, xi: float, tol: float = 10e-5, max_iter: int = 100) -> np.array:
    n = matrix.shape[0]
    matrix_xi = xi * matrix
    x = np.ones((n, 1))
    t = (x - xi) / n
    for i in range(max_iter):
        xnew = matrix_xi@x + t
        xnew = xnew / np.linalg.norm(xnew, 1)
        if np.linalg.norm(xnew - x, 1) < tol:
            break
        x = xnew
    return xnew


def hits_ii(test_pos: np.array, keep_value: float = 0):
    count_votes = np.sum(test_pos != 0, axis=2)
    for_votes = np.sum((test_pos < 0), axis=2)
    vote_matrix = for_votes / (count_votes + 1)
    if keep_value:
        vote_matrix = np.where(vote_matrix >= keep_value, vote_matrix, 0)
    gram_matrix = vote_matrix.T@vote_matrix
    ratings = hits_power_iteration(gram_matrix, 0.1)
    return partial_order(ratings)


def hits_ii_compute(train_users: list, test_users: list, train_test_same: bool = False, keep_value: float = 0):
    proposed_pos = []
    train_users_subset = train_users
    for i, tu in enumerate(test_users):
        _, _, hite_1, _, _, _, tite_1, _ = tu
        if train_test_same:
            train_users_subset = train_users[:i] + train_users[i+1:]
        test_pos = np.dstack([votes_on_items(tru[0], hite_1, tite_1, tru[1], tru[3]) for tru in train_users_subset])
        proposed_po = hits_ii(test_pos, keep_value=keep_value)
        proposed_pos.append(proposed_po)
    return proposed_pos
