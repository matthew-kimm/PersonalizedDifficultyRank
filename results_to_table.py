import pandas as pd
import argparse


def generate_table(dataset: str, repeats: int, seed: int, pct: float):
    round_score = 3
    round_time = 1
    
    table_head = """
    \\begin{table}[]
    \\small
    \\centering
    \\begin{tabular}{|c||c|c|c|c|c||c|c|c|c|}
        \\hline
        Ranking: & Set Split: & \\multicolumn{4}{|c|}{Per User Train/Test} & 
          \\multicolumn{4}{|c|}{Split Users Train/Test}  \\\\
         \\hline
         \\hline
          & \\textbf{Method} & \\textbf{AP} & \\textbf{AP\dag} & \\textbf{NDPM} & \\textbf{Time (s)} & \\textbf{AP} &
           \\textbf{AP\dag} & \\textbf{NDPM} & \\textbf{Time (s)} \\\\
         \\hline
         \\hline
    """

    easy_dataset_all = pd.read_csv(f'results/Experiment-Result__data-set_{dataset}__how-split_all__easiest.csv')\
        .set_index('METHOD')

    easy_dataset_pct = pd.read_csv(f'results/Experiment-Result__data-set_{dataset}__how-split_percent__'
                                   f'repeats_{repeats}__seed_{seed}__pct-train_{pct}__easiest.csv')
    easy_dataset_pct = easy_dataset_pct.groupby('METHOD').agg(
        TIME=pd.NamedAgg('TIME', 'mean'),
        AP_SCORE=pd.NamedAgg('AP_SCORE', 'mean'),
        NDPM_SCORE=pd.NamedAgg('NDPM_SCORE', 'mean'),
        REVERSE_AP_SCORE=pd.NamedAgg('REVERSE_AP_SCORE', 'mean')
    )

    hard_dataset_all = pd.read_csv(f'results/Experiment-Result__data-set_{dataset}__how-split_all__hardest.csv')\
        .set_index('METHOD')

    hard_dataset_pct = pd.read_csv(f'results/Experiment-Result__data-set_{dataset}__how-split_percent__'
                                   f'repeats_{repeats}__seed_{seed}__pct-train_{pct}__hardest.csv')
    hard_dataset_pct = hard_dataset_pct.groupby('METHOD').agg(
        TIME=pd.NamedAgg('TIME', 'mean'),
        AP_SCORE=pd.NamedAgg('AP_SCORE', 'mean'),
        NDPM_SCORE=pd.NamedAgg('NDPM_SCORE', 'mean'),
        REVERSE_AP_SCORE=pd.NamedAgg('REVERSE_AP_SCORE', 'mean')
    )

    methods = easy_dataset_pct.shape[0]
    method_map = {'HITS-II': 'HITS-II',
                  'HITS-IIT': 'HITS-IIT',
                  'EDURANK': 'EDURANK',
                  'EDURANK-AP-REVERSE': 'EDURANK-AP\dag',
                  'EDURANK-AP-SYMMETRIC': 'EDURANK-AP\ddag',
                  'EDURANK-NO-AP': 'EDURANK$\\neg$AP',
                  'RANDOM': 'RANDOM'}
    table_data = ''
    table_data += f"\\multirow{{{methods}}}{{*}}{{Easiest}}"
    for i in range(methods):
        method = easy_dataset_pct.index[i]
        # Easy
        table_data += f""" & \\textit{{{method_map[method]}}} & {round(easy_dataset_all.loc[method]['AP_SCORE'], round_score)} & {round(easy_dataset_all.loc[method]['REVERSE_AP_SCORE'], round_score)} &
          {round(easy_dataset_all.loc[method]['NDPM_SCORE'], round_score)} & {round(easy_dataset_all.loc[method]['TIME'], round_time)} & 
          {round(easy_dataset_pct.loc[method]['AP_SCORE'], round_score)} & {round(easy_dataset_pct.loc[method]['REVERSE_AP_SCORE'], round_score)}
           & {round(easy_dataset_pct.loc[method]['NDPM_SCORE'], round_score)} & 
          {round(easy_dataset_pct.loc[method]['TIME'], round_time)} \\\\
         """
        table_data += "\n\\hline\\hline\n" if i == (methods - 1) else "\n\\cline{2-10}\n"

    table_data += f"\\multirow{{{methods}}}{{*}}{{Hardest}}"
    for i in range(methods):
        method = easy_dataset_pct.index[i]
        # Hard
        table_data += f"""
          & \\textit{{{method_map[method]}}} & {round(hard_dataset_all.loc[method]['AP_SCORE'], round_score)} & {round(hard_dataset_all.loc[method]['REVERSE_AP_SCORE'], round_score)} &
         {round(hard_dataset_all.loc[method]['NDPM_SCORE'], round_score)} & {round(hard_dataset_all.loc[method]['TIME'], round_time)} & 
         {round(hard_dataset_pct.loc[method]['AP_SCORE'], round_score)} & {round(hard_dataset_pct.loc[method]['REVERSE_AP_SCORE'], round_score)}
         & {round(hard_dataset_pct.loc[method]['NDPM_SCORE'], round_score)} & 
         {round(hard_dataset_pct.loc[method]['TIME'], round_time)} \\\\
         """
        table_data += "\n\\hline\\hline\n" if i == (methods - 1) else "\n\\cline{2-10}\n"

    table_tail = f"""
      \\end{{tabular}}
      \\caption{{Experiment results for {dataset} data set.}}
      \\label{{tab:{dataset}}}
      \\end{{table}}
      """

    table = table_head + table_data + table_tail
    return table


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="Difficulty Rank Experiment LaTeX Table Generation",
        description="Generates results (ap, ndpm, time) in tabular form "
                    "within 2x2 (difficulty rank x train/test split method) outer table.",
        epilog="-------"
    )
    parser.add_argument('dataset', type=str, help="name of dataset, e.g. algebra or course")
    parser.add_argument('-r', '--repeats', type=int,
                        help="number of randomized repeats used in the experiment", default=1)
    parser.add_argument('-s', '--seed', type=int, help="starting integer seed used in the experiment", default=0)
    parser.add_argument('-p', '--percent_train', type=float, help="percentage of data for train, value in (0,1), " +
                                                                  "used in the experiment",
                        default=0.8)

    args = parser.parse_args()

    result_table = generate_table(args.dataset, args.repeats, args.seed, args.percent_train)

    with open(f'tables/{args.dataset}.tex', 'w') as f:
        f.write(result_table)
