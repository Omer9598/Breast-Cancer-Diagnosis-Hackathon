import pandas as pd
from functools import partial

# lymph column
LYMPH_MARKS = ['N0', 'NX', 'N1', 'N1a', 'N1b', 'N1c', 'N1mic', 'ITC', 'N2', 'N2a', 'N2b', 'N2c', 'N3', 'N3a', 'N3b', 'N3c', 'N4']
LYMPH_PREFIX = 'lymph'

# metastases column
METASTASES_MARKS = ['M0', 'MX', 'M1', 'M1a', 'M1b']
METASTASES_PREFIX = 'metastases'

TUMOR_MARKS = ['T0', 'TX', 'T1', 'T1a', 'T1b', 'T1c', 'T1mic', 'T1d', 'T2', 'T2a', 'T2b', 'T2c', 'T2d', 'T3',
               'T3a', 'T3b', 'T3c', 'T3d', 'T4', 'T4a', 'T4b', 'T4c', 'T4d', 'MF', 'Tis']
TUMOR_PREFIX = 'tumor'


def replace_na(marks, s):
    if s not in marks:
        return 'NA'
    return s


def preprocess_mark(df: pd.DataFrame, col, possible_marks, prefix):
    df[col] = df[col].apply(partial(replace_na, possible_marks))
    df = pd.get_dummies(df, prefix=prefix, columns=[col])
    for mark in possible_marks:
        if f'{prefix}_{mark}' not in df:
            df[f'{prefix}_{mark}'] = 0
    return df


def preprocess_lymph_node_mark(df: pd.DataFrame):
    return preprocess_mark(df, 'n_lymph_nodes_mark_(tnm)', LYMPH_MARKS, LYMPH_PREFIX)


def preprocess_tumor_node_mark(df: pd.DataFrame):
    return preprocess_mark(df, 't_tumor_mark_(tnm)', TUMOR_MARKS, TUMOR_PREFIX)


def preprocess_metastases_mark(df: pd.DataFrame):
    """
    This function encodes the metastases dynamically
    :param df: DataFrame to process
    :return: processed DF
    """
    return preprocess_mark(df, 'm_metastases_mark_(tnm)', METASTASES_MARKS, METASTASES_PREFIX)
