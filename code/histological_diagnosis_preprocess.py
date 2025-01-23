import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


def preprocess_histological_diagnosis(df: pd.DataFrame):
    """
    This function encodes a column dynamically - should be used on train only, need to extract columns
    :param df: DataFrame to preprocess
    :return: preprocessed DF
    """
    df['histological_diagnosis'] = df['histological_diagnosis'].apply(lambda s: s.replace('"', '').replace("AND ", "").
                                                                      replace("OF ", "").replace(",", "").
                                                                      replace("WITH ", "").replace("IN SITU ", "").replace("IN ", ""))
    df['histological_diagnosis_split'] = df['histological_diagnosis'].str.split()
    mlb = MultiLabelBinarizer()
    df = df.join(pd.DataFrame(mlb.fit_transform(df.pop('histological_diagnosis_split')), columns=mlb.classes_, index=df.index))

    return df
