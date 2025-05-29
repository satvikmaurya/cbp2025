import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import numpy as np
import os
from collections import Counter
from matplotlib import pyplot as plt
from typing import List
import ray

def load_correlations(file):
    df = pd.read_csv(file)
    K = 32
    records = []
    for i, row in df[df['inst_type'] == 0].iterrows():
        if i < 1000:
            continue
        window = df.iloc[max(0, i - K):i]
        loads = window[window['inst_type'] == 1]
        last = loads.tail(5)['address'].tolist()
        rec = {
            'pc': int(row['pc'], 16),
            'mispred': row['predicted'] != row['actual'],
            'dist_last_load': i - loads.index.max() if not loads.empty else K+1,
            'load_addr0': int(last[-1], 16) if len(last) > 0 else 'none',
            'load_addr1': int(last[-2], 16) if len(last) > 1 else 'none',
            'load_addr2': int(last[-3], 16) if len(last) > 2 else 'none',
            'load_addr3': int(last[-4], 16) if len(last) > 3 else 'none',
            'load_addr4': int(last[-5], 16) if len(last) > 4 else 'none',
        }
        rec['xor_pc_load0'] = rec['pc'] ^ (rec['load_addr0'] if rec['load_addr0']!='none' else 0)
        records.append(rec)
        pass
    return records

if __name__ == "__main__":
    file = './tagescl/fp_0_trace_branch_misps.csv'

    records = load_correlations(file)
    df = pd.DataFrame.from_records(records)
    
    brdf = df

    cat_cols = ['load_addr0','load_addr1','load_addr2', 'load_addr3','load_addr4']
    num_cols = ['dist_last_load','xor_pc_load0']
    X = brdf[cat_cols + num_cols]
    y = brdf['mispred']

    enc = OneHotEncoder(handle_unknown='ignore')
    ct  = ColumnTransformer([('cat',enc,cat_cols)], remainder='passthrough')
    Xenc = ct.fit_transform(X)
    mi = mutual_info_classif(Xenc, y, discrete_features=True)
    print(sorted(zip(ct.get_feature_names_out(), mi), key=lambda x:-x[1])[:20])