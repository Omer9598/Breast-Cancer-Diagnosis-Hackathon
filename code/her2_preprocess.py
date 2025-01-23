def check_her2_gene_text(text):
    text = text.replace(' ', '').lower()
    no_digit_text = ''.join([char for char in str(text) if not char.isdigit()])
    only_digit = ''.join([char for char in str(text) if char.isdigit()])

    if 'pos' in no_digit_text:
        return 1
    if 'חיובי' in no_digit_text:
        return 1

    if 'fish+' in no_digit_text:
        return 1

    if 'jhuch' in no_digit_text:
        return 1

    if 'amplified' in no_digit_text and 'non' not in no_digit_text and 'notamplified' not in text and 'noamplified' not in text:
        return 1

    if len(only_digit) > 0 and int(only_digit[0]) == 3:
        return 1

    if text == '+':
        return 1

    if text == '100':
        return 1
    if '+3' in text:
        return 1

    if '(+)' == text:
        return 1

    if len(set(text)) == 1 and '+' in set(text):
        return 1

    return 0


def process_her2(df):
    df['her2'].fillna('neg', inplace=True)
    df['her2_gene_exists'] = df['her2'].apply(check_her2_gene_text)
    df.drop('her2', axis=1, inplace=True)

    return df
