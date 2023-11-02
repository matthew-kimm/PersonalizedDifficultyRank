import pandas as pd
from os import mkdir, getcwd, chdir
from os.path import exists, dirname


if __name__ == '__main__':

    prev_cwd = getcwd()

    if dirname(__file__) != getcwd():
        chdir(dirname(__file__))

    if not exists('data_information'):
        mkdir('data_information')

    if not exists('transformed_data'):
        mkdir('transformed_data')

    column_map = {'Anon Student Id': 'user', 'Problem Name': 'item', 'Step Duration (sec)': 'duration',
                  'Correct First Attempt': 'correct_first_attempt', 'Incorrects': 'incorrects', 'Step End Time': 'time'}
    columns = list(column_map.keys())
    # The train/test were for a KDD Cup Task, we are merging into one dataset for our own split later
    train = pd.read_csv('algebra_2005_2006_train.txt', sep='\t', usecols=columns)
    test = pd.read_csv('algebra_2005_2006_test.txt', sep='\t', usecols=columns)
    train = train.rename(columns=column_map)
    test = test.rename(columns=column_map)
    df = pd.concat([train, test])
    df['time'] = pd.to_datetime(df['time'])

    # Information: number of times each user had each item before combining duplicates
    df.groupby(['user', 'item']).agg(
        count=pd.NamedAgg('time', 'count')
    ).to_csv('data_information/user_attempt_item_count.csv', index=True)

    # Count before drop
    with open('data_information/count_before_drop.txt', 'w') as f:
        f.write(f"user: {df['user'].nunique()}, item: {df['item'].nunique()}")

    df = df.groupby(['user', 'item']).agg(
        correct_first_attempt=pd.NamedAgg('correct_first_attempt', 'mean'),
        incorrects=pd.NamedAgg('incorrects', 'mean'),
        duration=pd.NamedAgg('duration', 'mean'),
        time=pd.NamedAgg('time', 'max'),
    ).reset_index()

    df['count'] = df.groupby('user')['time'].transform('count')

    df = df[df['count'] >= 6]

    # Information: total item count per user
    df.groupby(['user']).agg(
        count=pd.NamedAgg('item', 'count')
    ).to_csv('data_information/user_unique_item_count.csv', index=True)

    df = df.sort_values(by=['user', 'correct_first_attempt', 'incorrects', 'duration'],
                        ascending=[False, False, True, True])
    current_user = ''
    rank_count = 1
    ranks = []
    for user in df['user']:
        if user == current_user:
            rank_count += 1
            ranks.append(rank_count)
        else:
            rank_count = 1
            current_user = user
            ranks.append(rank_count)

    # Rating is currently from easiest (1) to most difficulty (n)
    df['rating'] = ranks

    df = df[['user', 'item', 'time', 'rating']]
    df = df.sort_values(by=['user', 'time'])

    # split into two even time sets
    df['mid_time'] = df.groupby(['user'])['time'].transform('median')
    df['time'] = (df['time'] > df['mid_time']).astype(int)

    # Information: how often item is in train/test split when all users part of both train/test
    item_time_split = df.groupby('item').agg(pct_in_test_set=pd.NamedAgg('time', 'mean'),
                                             count=pd.NamedAgg('time', 'count'))
    item_time_split['pct_in_train_set'] = 1 - item_time_split['pct_in_test_set']
    item_time_split = item_time_split[['pct_in_train_set', 'pct_in_test_set', 'count']]
    item_time_split['pct_test_between_25_and_75'] = ((item_time_split['pct_in_test_set'] >= 0.25) &
                                                     (item_time_split['pct_in_test_set'] <= 0.75)).astype(int)
    item_time_split.to_csv('data_information/item_percent_time_split.csv', index=True)
    # assign user/item ids
    df['user'] = df['user'].map({v: k for k, v in enumerate(df['user'].unique())})
    df['item'] = df['item'].map({v: k for k, v in enumerate(df['item'].unique())})

    # Information: total user/item count
    pd.DataFrame({'user_count': df['user'].nunique(), 'item_count': df['item'].nunique()}, index=[0])\
        .to_csv('data_information/overall_unique_user_item_count.csv', index=False)

    # Rating Easiest to Hardest (highest value)
    df = df[['user', 'item', 'time', 'rating']]
    df.to_csv('transformed_data/algebra_easiest.csv', index=False)

    # Rating Hardest to Easiest (highest value)
    df['rating'] = df['rating'].max() + 1 - df['rating']
    df.to_csv('transformed_data/algebra_hardest.csv', index=False)

    chdir(prev_cwd)
