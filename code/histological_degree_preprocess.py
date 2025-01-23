import pandas


def handle_histological_degree(text: str) -> int:
    """
    This function will convert the values in the histological degree feature
    G1-4 will be converted to ints
    GX, null will be 0
    :param text: the text of the entry
    :return: the corresponding int
    """
    # G1-4
    for number in range(1, 4):
        if text[1] == str(number):
            return number

    # GX, null values
    return 0


def preprocess_histological_degree(df: pandas.DataFrame) -> pandas.DataFrame:
    """
    handling the whole column
    :param df: the database df
    :return: preprocessed df
    """
    df['histopatological_degree'] = df['histopatological_degree'].apply(
        lambda x: handle_histological_degree(x))
    return df
