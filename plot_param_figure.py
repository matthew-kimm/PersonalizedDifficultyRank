import argparse
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_figure(repeats: int, seed: int, pct: float):
    param_dataset = pd.read_csv(f'results/Experiment-Result__data-set_algebra__how-split_percent__'
                                   f'repeats_{repeats}__seed_{seed}__pct-train_{pct}__easiest__param.csv')
    param_dataset['PARAM'] = param_dataset['METHOD'].apply(lambda x: float(x.split('-')[-1]))
    param_dataset = param_dataset.drop(columns=['METHOD'])
    param_dataset = param_dataset.groupby('PARAM').agg(
        TIME=pd.NamedAgg('TIME', 'mean'),
        AP_SCORE=pd.NamedAgg('AP_SCORE', 'mean'),
        REVERSE_AP_SCORE=pd.NamedAgg('REVERSE_AP_SCORE', 'mean'),
        NAN_REVERSE_AP_SCORE=pd.NamedAgg('NAN_REVERSE_AP_SCORE', 'mean'),
        NDPM_SCORE=pd.NamedAgg('NDPM_SCORE', 'mean')
    )

    font = {'weight': 'bold',
            'size': 18}

    matplotlib.rc('font', **font)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    df = param_dataset[['REVERSE_AP_SCORE', 'AP_SCORE', 'NDPM_SCORE', 'NAN_REVERSE_AP_SCORE']].copy()
    # TODO: keep track of test size, currently manually set denominator (test size)
    df['NAN_REVERSE_AP_SCORE'] = df['NAN_REVERSE_AP_SCORE'] / 107

    ap_score = ax.plot(df.index, df['AP_SCORE'], '-.', label='AP_SCORE', color='orange')
    ndpm_score = ax.plot(df.index, df['NDPM_SCORE'], 'g-.', label='NDPM_SCORE')
    nan_reverse_ap_score = ax.plot(df.index, df['NAN_REVERSE_AP_SCORE'], 'r-.', label='NAN_AP$\dag$_PROPORTION')
    ax_new = ax.twinx()
    reverse_ap_score = ax_new.plot(df.index, df['REVERSE_AP_SCORE'], 'b-*', label='AP$\dag$_SCORE (right)')
    ax.set_title('HITS-IIT AP$\dag$ Score by Threshold', fontdict=font)
    ax.set_xlabel('Threshold', fontdict=font)
    ax.set_ylabel('Other Score', fontdict=font)
    ax_new.set_ylabel('AP$\dag$ Score', fontdict=font)
    ax_new.set_yticks(np.arange(0.60, 0.72, 0.02))

    lines = ap_score + ndpm_score + nan_reverse_ap_score + reverse_ap_score
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels, prop={'size': 12})

    plt.savefig('figures/param.pdf', bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="HITS-IIT Parameter Plot",
        description="Generates plot for reverse ap score (right y-axis) and "
                    "other scores (ndpm, ap, proportion reverse ap nan)",
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
