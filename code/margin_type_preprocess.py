import pandas
import pandas as pd


def handle_margin_type(text: str) -> int:
    """
    This function will convert the values in the margin type feature column:
    0 - negative/ no
    1 - yes
    :param text: the text to convert
    :return: the corresponding number
    """
    if text == 'נגועים':
        return 1
    return 0


def preprocess_margin_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function will process the margin_type feature in the dataframe
    :param df: DataFrame to preprocess
    :return: processed DF
    """
    df['margin_type'] = df['margin_type'].apply(
        lambda x: handle_margin_type(x))
    return df
