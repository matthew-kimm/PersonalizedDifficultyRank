import numpy as np
import pandas as pd
from partial_order import partial_order
from edurank import edurank_compute
from hits_ii import hits_ii_compute
from metrics import ap_score_how, ndpm_score_how
import time
from functools import lru_cache, partial
import logging
import argparse
from os.path import join as path_join


def random_compute(train_users: list, test_users: list):
    rpos = []
    for i, tu in enumerate(test_users):
        _, _, _, _, _, _, _, tn = tu
        rpo = partial_order(np.random.rand(tn, 1))
        rpos.append(rpo)
    return rpos


@lru_cache()
def read_data(path_to_file: str):
    logging.info(f'Attempting to read data from {path_to_file}.')

    with open(path_to_file, 'r') as f:
        logging.info(f'Checking data headers.')
        x = f.readlines(1)[0].splitlines()[0].split(',')
        if x != ['user', 'item', 'time', 'rating']:
            raise Exception('Columns must be (in order): user,item,time,rating')
    logging.info('Headers checked.')

    logging.info('Reading data.')
    arr = np.loadtxt(path_to_file, delimiter=',', skiprows=1, dtype=float)
    logging.info(f'Data read, {arr.shape[0]} observations.')

    logging.info('Verifying user ids (integer) start at 0 and '
                 'do not skip (i.e. some user k exists but not user k-1 does not).')
    unique_users = int(np.max(arr[:, 0])) + 1
    if not np.array_equal(np.unique(arr[:, 0]), np.arange(unique_users)):
        raise Exception('User ids (integer) must start at 0 and not skip ( for every user k != 0, user k-1 must exist. )')
    logging.info(f'Verified, {unique_users} unique users.')

    logging.info('Verifying item ids (integer) start at 0 and '
                 'do not skip (i.e. some item k exists but not item k-1 does not).')
    unique_items = int(np.max(arr[:, 1])) + 1
    if not np.array_equal(np.unique(arr[:, 1]), np.arange(unique_items)):
        raise Exception('Item ids (integer) must start at 0 and not skip ( for every item k != 0, item k-1 must exist. )')

    logging.info(f'Verified, {unique_items} unique items.')

    return arr, unique_users, unique_items


def initialize_data(path_to_file: str, random_seed: int = 0, percent: float = 0):
    """
    Train/Test split as user (time=0) in train and same user (time=1) in test
    :param path_to_file:
    :param random_seed:
    :param percent:
    :return:
    """
    logging.info('---LOAD DATA---')
    arr, unique_users, unique_items = read_data(path_to_file)
    logging.info('---LOAD DATA (Complete)---')

    np.random.seed(random_seed)
    logging.info(f'Set seed to {random_seed}.')

    if percent > 0:
        logging.info(f'Creating Train/Test Split using {percent * 100}% train.')
    else:
        logging.info(f'Creating Train/Test Split every user in train and test, '
                     f'train with items at time (t=0) and test with items at (t=1).')

    users = np.arange(unique_users)
    train_count = int(unique_users * percent)
    train_users = np.random.choice(users, train_count, replace=False)
    test_users = users[~np.isin(users, train_users)]

    # sort user, item, time, rating
    arr = arr[np.lexsort((arr[:, 3], arr[:, 1], arr[:, 2], arr[:, 0])), :]
    users = arr[:, 0]
    user_splits = np.where(users[:-1] != users[1:])[0] + 1

    user_data = np.vsplit(arr, user_splits)

    train = []
    test = []
    for i, ud in enumerate(user_data):
        if percent > 0 and i in train_users:
            history_arr = ud[:, [1, 3]]
        else:
            history_arr = ud[np.where(ud[:, 2] == 0)[0], :][:, [1, 3]]
        history_items = history_arr[:, 0].tolist()
        history_ratings = history_arr[:, [1]]
        history_map = {it: i for i, it in enumerate(history_items)}
        hn = len(history_ratings)

        target_arr = ud[np.where(ud[:, 2] == 1)[0], :][:, [1, 3]]
        target_items = target_arr[:, 0].tolist()
        target_ratings = target_arr[:, [1]]
        target_map = {it: i for i, it in enumerate(target_items)}
        tn = len(target_ratings)

        history_partial_order = partial_order(history_ratings)
        target_partial_order = partial_order(target_ratings)

        if percent > 0:
            if i in train_users:
                train.append((history_partial_order, history_map, history_items, hn))
            else:
                test.append((history_partial_order, history_map, history_items, hn,
                             target_partial_order, target_map, target_items, tn))
        else:
            train.append((history_partial_order, history_map, history_items, hn))
            test.append((history_partial_order, history_map, history_items, hn,
                         target_partial_order, target_map, target_items, tn))

    logging.info(f'Resulting Train/Test Split: Train size {len(train)}, Test Size {len(test)}.')

    return train, test


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser(
        prog="Difficulty Rank Experiment",
        description="Conducts experiment using EduRank and HITS-II.",
        epilog="-------"
    )
    parser.add_argument('dataset', type=str, help="name of dataset, e.g. algebra or course (if available),\n"
                                                  "data must have columns (in order): user, item, time, rating "
                                                  "where rating is the rating value "
                                                  "(higher value => better rank) unless -hardest set then "
                                                  "(lower value => better rank)\n"
                                                  "user ids must be from 0 to # of users.\n"
                                                  "item ids must be from 0 to # of items.\n"
                                                  "time should be either 0 or 1.")
    parser.add_argument('how_split', type=str, help="e.g. all or percent,\n"
                                                    "all puts user in both train (time=0) and test (time=1)\n"
                                                    "percent puts a percent of users in train (time=0,1)"
                                                    "and remaining users in test (time=0,1)",
                        default='percent')
    parser.add_argument('-r', '--repeats', type=int,
                        help="number of randomized repeats of the experiment (percent split only)",
                        default=1)
    parser.add_argument('-s', '--seed', type=int, help="starting integer seed (percent split only)",
                        default=0)
    parser.add_argument('-p', '--percent_train', type=float, help="percentage of data for train, value in (0,1), " +
                                                                  "(percent split only)",
                        default=0.8)
    parser.add_argument('-hardest', action='store_true', help="this flag ranks by hardest first, "
                                                              "default is easiest first")
    parser.add_argument('-param', action='store_true', help="only run parameter analysis for HITS-IIT "
                                                            "(use percent split)")
    args = parser.parse_args()

    result_data = []
    result_columns = ['SEED_OFFSET', 'METHOD', 'TIME',
                      'AP_SCORE', 'NAN_AP_SCORE',
                      'REVERSE_AP_SCORE', 'NAN_REVERSE_AP_SCORE',
                      'SYMMETRIC_AP_SCORE', 'NAN_SYMMETRIC_AP_SCORE',
                      'NDPM_SCORE', 'NAN_NDPM_SCORE',
                      'REVERSE_NDPM_SCORE', 'NAN_REVERSE_NDPM_SCORE',
                      'SYMMETRIC_NDPM_SCORE', 'NAN_SYMMETRIC_NDPM_SCORE']

    data_path = path_join('data', f'{args.dataset}', 'transformed_data',
                          f'{args.dataset}_{"easiest" if not args.hardest else "hardest"}.csv')

    repeats = args.repeats if args.how_split == 'percent' else 1

    for repeat in range(repeats):
        logging.info(f'Test Run: {repeat + 1} / {repeats}')
        if args.how_split == 'percent':
            seed = args.seed + repeat
            train_data, test_data = initialize_data(data_path, random_seed=seed, percent=args.percent_train)
        elif args.how_split == 'all':
            train_data, test_data = initialize_data(data_path)
        else:
            raise Exception("how_split must be either percent or all.")

        train_test_same = args.how_split == 'all'
        logging.info(f'Run {repeat + 1} / {repeats}')

        hits_ii_method = partial(hits_ii_compute, train_test_same=train_test_same)
        # TODO: allow user input / config value, currently manually set threshold value
        hits_iit_method = partial(hits_ii_compute, keep_value=0.8)
        edurank_method = partial(edurank_compute, train_test_same=train_test_same)
        edurank_ap_reverse_method = partial(edurank_compute, how='reverse', train_test_same=train_test_same)
        edurank_ap_symmetric_method = partial(edurank_compute, how='symmetric', train_test_same=train_test_same)
        edurank_no_ap = partial(edurank_compute, use_ap=False, train_test_same=train_test_same)

        if args.param:
            name_and_method = {f'HITS-IIT-{i}': partial(hits_ii_compute, keep_value=i)
                               for i in [round(j, 2) for j in np.arange(0, 1.05, 0.05)]}
        else:
            name_and_method = {'HITS-II': hits_ii_method,
                               'HITS-IIT': hits_iit_method,
                               'EDURANK': edurank_method,
                               'EDURANK-AP-REVERSE': edurank_ap_reverse_method,
                               'EDURANK-AP-SYMMETRIC': edurank_ap_symmetric_method,
                               'EDURANK-NO-AP': edurank_no_ap,
                               'RANDOM': random_compute}
        for method, compute in name_and_method.items():
            logging.info(f'Computing {method}.')
            start = time.time()
            ppos = compute(train_data, test_data)
            end = time.time()
            run_time = end - start
            logging.info(f'Computed {method} in {int(run_time)} seconds.')

            test_ap_scores = np.array([ap_score_how(test_data[i][4], ppo) for i, ppo in enumerate(ppos)])
            test_reverse_ap_scores = np.array([ap_score_how(test_data[i][4], ppo, how='reverse') for i, ppo in enumerate(ppos)])
            test_symmetric_ap_scores = np.array([ap_score_how(test_data[i][4], ppo, how='symmetric') for i, ppo in enumerate(ppos)])

            test_ndpm_scores = np.array([ndpm_score_how(test_data[i][4], ppo) for i, ppo in enumerate(ppos)])
            test_reverse_ndpm_scores = np.array([ndpm_score_how(test_data[i][4], ppo, how='reverse') for i, ppo in enumerate(ppos)])
            test_symmetric_ndpm_scores = np.array([ndpm_score_how(test_data[i][4], ppo, how='symmetric') for i, ppo in enumerate(ppos)])

            result_ap_score = np.nanmean(test_ap_scores)
            result_nan_ap_score = np.count_nonzero(np.isnan(test_ap_scores))
            result_reverse_ap_score = np.nanmean(test_reverse_ap_scores)
            result_nan_reverse_ap_score = np.count_nonzero(np.isnan(test_reverse_ap_scores))
            result_symmetric_ap_score = np.nanmean(test_symmetric_ap_scores)
            result_nan_symmetric_ap_score = np.count_nonzero(np.isnan(test_symmetric_ap_scores))

            result_ndpm_score = np.nanmean(test_ndpm_scores)
            result_nan_ndpm_score = np.count_nonzero(np.isnan(test_ndpm_scores))
            result_reverse_ndpm_score = np.nanmean(test_reverse_ndpm_scores)
            result_nan_reverse_ndpm_score = np.count_nonzero(np.isnan(test_reverse_ndpm_scores))
            result_symmetric_ndpm_score = np.nanmean(test_symmetric_ndpm_scores)
            result_nan_symmetric_ndpm_score = np.count_nonzero(np.isnan(test_symmetric_ndpm_scores))

            datum = [repeat, method, run_time,
                     result_ap_score, result_nan_ap_score,
                     result_reverse_ap_score, result_nan_reverse_ap_score,
                     result_symmetric_ap_score, result_nan_symmetric_ap_score,
                     result_ndpm_score, result_nan_ndpm_score,
                     result_reverse_ndpm_score, result_nan_reverse_ndpm_score,
                     result_symmetric_ndpm_score, result_nan_symmetric_ap_score]
            result_data.append(datum)

    result_df = pd.DataFrame(result_data, columns=result_columns)
    print(result_df.drop(columns=['SEED_OFFSET']).groupby('METHOD').agg('mean'))
    if args.how_split == 'all':
        result_name_tail = f''
    elif args.how_split == 'percent':
        result_name_tail = f'__repeats_{args.repeats}__seed_{args.seed}__pct-train_{args.percent_train}'

    if args.param:
        result_df.to_csv(f'results/Experiment-Result__data-set_{args.dataset}__how-split_{args.how_split}'
                         f'{result_name_tail}__{"hardest" if args.hardest else "easiest"}__param.csv', index=False)
    else:
        result_df.to_csv(f'results/Experiment-Result__data-set_{args.dataset}__how-split_{args.how_split}'
                         f'{result_name_tail}__{"hardest" if args.hardest else "easiest"}.csv', index=False)
