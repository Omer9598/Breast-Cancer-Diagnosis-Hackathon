import numpy as np


def handle_ki67_text(text):
    if not text or text == '':
        return 0
    text = text.replace(' ', '').lower()
    text = text.replace('%', '')
    simple_decimal_result = float(text) if text.isdecimal() else None

    if simple_decimal_result:
        if simple_decimal_result < 0:
            simple_decimal_result = 0
        if simple_decimal_result > 100:
            simple_decimal_result = 100

        return simple_decimal_result / 100

    # no simple decimal result, looking for numbers range
    if '-' in text:
        splitted = text.split('-')
        numbers = [float(number_text) if number_text.isdecimal() else None for number_text in splitted]
        numbers = [number for number in numbers if number is not None]
        numbers = [number for number in numbers if 0 <= number <= 100]
        if len(numbers) > 1:
            return np.mean(numbers) / 100
    return 0


def preprocess_ki67_protien(df):
    df['ki67_protein'].fillna('0', inplace=True)
    df['ki67_protein'] = df['ki67_protein'].apply(handle_ki67_text)
    return df
