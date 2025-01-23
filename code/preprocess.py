from .her2_preprocess import process_her2
from .basic_stage_preprocess import preprocess_basic_stage
from .ki67_protien_preprocess import preprocess_ki67_protien
from .histological_diagnosis_preprocess import preprocess_histological_diagnosis
from .histological_degree_preprocess import preprocess_histological_degree
from .marks_preprocess import preprocess_lymph_node_mark, preprocess_metastases_mark, preprocess_tumor_node_mark
from .margin_type_preprocess import preprocess_margin_type

import pandas as pd


def load_data(feature_path, label_path):
    features = pd.read_csv(feature_path)
    labels = pd.read_csv(label_path)
    return map_feature_columns(pd.concat([features, labels], axis=1))


def map_feature_columns(df):
    df.columns = df.columns.str.replace('אבחנה', '')
    df.columns = df.columns.str.replace('-', '')
    df.columns = df.columns.str.replace(' ', '_')
    df.columns = df.columns.str.lower()
    return df


def preprocess(X, y=None):
    X = process_her2(X)
    X = preprocess_basic_stage(X)
    X = preprocess_ki67_protien(X)
    X = preprocess_histological_diagnosis(X)
    X = preprocess_histological_degree(X)
    X = preprocess_lymph_node_mark(X)
    X = preprocess_metastases_mark(X)
    X = preprocess_tumor_node_mark(X)
    X = preprocess_margin_type(X)
    return X, y
