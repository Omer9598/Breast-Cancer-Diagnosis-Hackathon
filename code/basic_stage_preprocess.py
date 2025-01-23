def handle_basic_stage(text):
    if text == 'n' or not text or text == '':
        return 0

    text = text.lower()
    if text.startswith('c'):
        # clinical stage
        return 1
    if text.startswith('p'):
        # pathological stage, post surgery
        return 2
    if text.startswith('r'):
        return 3

    return 0


def preprocess_basic_stage(df):
    df['basic_stage'].fillna('n', inplace=True)
    df['basic_stage'] = df['basic_stage'].apply(lambda x: handle_basic_stage(x))
    return df