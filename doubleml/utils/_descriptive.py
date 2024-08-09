import numpy as np
import pandas as pd


def generate_summary(coef, se, t_stat, pval, ci, index_names=None):
    col_names = ['coef', 'std err', 't', 'P>|t|']
    summary_stats = np.transpose(np.vstack(
        [coef, se, t_stat, pval]))
    df_summary = pd.DataFrame(summary_stats, columns=col_names)
    if index_names is not None:
        df_summary.index = index_names
    df_summary = df_summary.join(ci)
    return df_summary
