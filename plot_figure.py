import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from typing import List


def autolabel(ax, rects, round_digits: int):
    for rect in rects:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., 1.005 * h, f'{round(h, round_digits)}',
                ha='center', va='bottom')


def group_bar_chart(datasets: List[pd.DataFrame], column: str, title: str = '', y_label: str = '',
                    group_labels: List[str] = None, round_digits: int = 2, fig_output: str = ''):
    """
    Plot column of interest in each dataset grouped by an element of index in each dataset
    :param datasets:
    :param column:
    :param title:
    :param y_label:
    :param group_labels:
    :param round_digits:
    :param fig_output:
    :return:
    """

    if group_labels is None:
        group_labels = [""] * len(datasets)

    font = {'weight': 'bold',
            'size': 18}

    matplotlib.rc('font', **font)

    n = len(datasets)
    grouping_offset = np.arange(0, 8*n, 8)
    idxs = datasets[0].index
    method_map = {'HITS-II': 'HITS-II',
                  'HITS-IIT': 'HITS-IIT',
                  'EDURANK': 'EDURANK',
                  'EDURANK-AP-REVERSE': 'EDURANK-AP$\dag$',
                  'EDURANK-AP-SYMMETRIC': 'EDURANK-AP$\ddag$',
                  'EDURANK-NO-AP': 'EDURANK$\\neg$AP',
                  'RANDOM': 'RANDOM'}
    mapped_idxs = [method_map[idx] for idx in idxs]
    m = len(idxs)
    width = 8 / (m + 0.2)

    # add more colors or implement colorwheel as argument to extend
    color_wheel = 'bgrcmyk' * 10

    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(111)

    vals_ = []
    rects_ = []

    for i, idx in enumerate(idxs):
        vals = [data.loc[idx][column] for data in datasets]
        vals_.append(vals)
        rects = ax.bar(grouping_offset + i * width, vals, width, color=color_wheel[i])
        rects_.append(rects)

    # Adds room for legend
    invis_rect = ax.bar(0, max([max(vs) for vs in vals_]) * 1.2, alpha=0.0)

    ax.set_ylabel(ylabel=y_label, fontdict=font)
    ax.set_xticks(grouping_offset + (width * m / 2))
    ax.set_xticklabels(tuple(group_labels))
    ax.legend(tuple([rects[0] for rects in rects_]),
              tuple(mapped_idxs))
    ax.set_title(title, fontdict=font)

    [autolabel(ax, rects=rects, round_digits=round_digits) for rects in rects_]

    plt.savefig(f'{fig_output}', bbox_inches='tight')


def plot_figure(repeats: int, seed: int, pct: float):
    easy_algebra_dataset_pct = pd.read_csv(f'results/Experiment-Result__data-set_algebra__how-split_percent__'
                                           f'repeats_{repeats}__seed_{seed}__pct-train_{pct}__easiest.csv')
    easy_algebra_dataset_pct = easy_algebra_dataset_pct.groupby('METHOD').agg(
        TIME=pd.NamedAgg('TIME', 'mean'),
        AP_SCORE=pd.NamedAgg('AP_SCORE', 'mean'),
        NDPM_SCORE=pd.NamedAgg('NDPM_SCORE', 'mean'),
        REVERSE_AP_SCORE=pd.NamedAgg('REVERSE_AP_SCORE', 'mean'),
    )

    easy_course_dataset_pct = pd.read_csv(f'results/Experiment-Result__data-set_course__how-split_percent__'
                                          f'repeats_{repeats}__seed_{seed}__pct-train_{pct}__easiest.csv')
    easy_course_dataset_pct = easy_course_dataset_pct.groupby('METHOD').agg(
        TIME=pd.NamedAgg('TIME', 'mean'),
        AP_SCORE=pd.NamedAgg('AP_SCORE', 'mean'),
        NDPM_SCORE=pd.NamedAgg('NDPM_SCORE', 'mean'),
        REVERSE_AP_SCORE = pd.NamedAgg('REVERSE_AP_SCORE', 'mean'),
    )

    group_bar_chart([easy_algebra_dataset_pct, easy_course_dataset_pct], 'TIME',
                    'Time (Sec) for EduRank vs HITS-II', 'Time (Sec)',
                    ['Algebra', 'Course'], 1, 'figures/time.pdf')

    group_bar_chart([easy_algebra_dataset_pct, easy_course_dataset_pct], 'AP_SCORE',
                    'AP Score for EduRank vs HITS-II', 'AP Score',
                    ['Algebra', 'Course'], 3, 'figures/ap.pdf')

    group_bar_chart([easy_algebra_dataset_pct, easy_course_dataset_pct], 'REVERSE_AP_SCORE',
                    'AP$\dag$ Score for EduRank vs HITS-II', 'AP$\dag$ Score',
                    ['Algebra', 'Course'], 3, 'figures/reverse_ap.pdf')

    group_bar_chart([easy_algebra_dataset_pct, easy_course_dataset_pct], 'NDPM_SCORE',
                    'NDPM Score for EduRank vs HITS-II', 'NDPM Score',
                    ['Algebra', 'Course'], 3, 'figures/ndpm.pdf')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="Difficulty Rank Experiment Plot Generation",
        description="Generates results (ap, ndpm, time) in bar chart form "
                    "for easiest difficulty ranking using percent split.",
        epilog="-------"
    )
    parser.add_argument('-r', '--repeats', type=int,
                        help="number of randomized repeats used in the experiment", default=1)
    parser.add_argument('-s', '--seed', type=int, help="starting integer seed used in the experiment", default=0)
    parser.add_argument('-p', '--percent_train', type=float, help="percentage of data for train, value in (0,1), " +
                                                                  "used in the experiment",
                        default=0.8)

    args = parser.parse_args()

    plot_figure(args.repeats, args.seed, args.percent_train)
