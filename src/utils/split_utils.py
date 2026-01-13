from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
import pandas as pd
import random

def stratified_split(train_df, split_col='label', test_size=0.2, seed=42):

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)

    for train_idx, val_idx in sss.split(train_df, train_df[split_col]):
        train_ids = train_df.iloc[train_idx]['track_id'].tolist()
        val_ids = train_df.iloc[val_idx]['track_id'].tolist()
        return train_ids, val_ids

def get_split_keys(keys, labels, test_size=0.2, seed=42, shuffle=False):

    df = pd.DataFrame({'track_id': keys, 'label': labels})

    trainval_keys, test_keys = stratified_split(df, split_col='label', test_size=test_size, seed=seed)

    trainval_labels = df[df['track_id'].isin(trainval_keys)]['label'].tolist()
    train_df = pd.DataFrame({'track_id': trainval_keys, 'label': trainval_labels})

    train_keys, val_keys = stratified_split(train_df, split_col='label', test_size=test_size, seed=seed+1)

    # Reorder train_keys to ensure a balanced distribution of labels
    if shuffle:
        random.seed(seed)
        random.shuffle(train_keys)

    return train_keys, val_keys, test_keys


def carn_split_keys(csv_path, split_col='Taala', test_size=0.2, seed=42, reorder=True):


    split_df = pd.read_csv(csv_path, dtype={'track_id': str})

    train_df, test_df = train_test_split(split_df, test_size=test_size, random_state=seed, stratify=split_df[split_col])

    train_keys, val_keys = stratified_split(train_df, split_col=split_col, test_size=test_size, seed=seed)

    test_keys = test_df.sort_values([split_col, 'track_id'])['track_id'].tolist()

    if reorder:
        train_keys = reorder_by_taala_proportion(train_df, train_keys)

    return train_keys, val_keys, test_keys


def reorder_by_taala_proportion(df, train_keys):

    train_df = df[df['track_id'].isin(train_keys)].copy()

    # Group track_ids by Taala
    grouped = train_df.groupby('Taala')['track_id'].apply(list).to_dict()

    # Sort track_ids for reproducibility
    for taala in grouped:
        grouped[taala].sort()

    # Use Pandas to count examples per Taala
    taala_counts = train_df['Taala'].value_counts()
    total = taala_counts.sum()
    taala_ratios = (taala_counts / total).to_dict()

    # Initialize trackers
    used_counts = {taala: 0 for taala in grouped}
    quotas = {taala: 0.0 for taala in grouped}
    reordered_keys = []

    for _ in range(total):
        # Update quotas based on ratios
        for taala in quotas:
            quotas[taala] += taala_ratios[taala]

        # Choose the Taala with max quota and remaining samples
        available = [(quota, taala) for taala, quota in quotas.items() if used_counts[taala] < taala_counts[taala]]
        _, chosen_taala = max(available)

        # Append the next track_id
        index = used_counts[chosen_taala]
        reordered_keys.append(grouped[chosen_taala][index])
        used_counts[chosen_taala] += 1
        quotas[chosen_taala] -= 1

    return reordered_keys
