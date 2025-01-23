from hackathon_code.TumorSizeRegressor import TumorSizeRegressor
from sklearn.model_selection import train_test_split
from hackathon_code.utils import *
from hackathon_code.config import COLS_TO_DROP, TRAIN_DATA_PATH, TRAIN_LABELS_PATH
from hackathon_code.preprocess import load_data, preprocess
from hackathon_code.eda_plots import *

def main_regression():
    label = 'tumor_size'
    df = load_data(
        TRAIN_DATA_PATH,
        TRAIN_LABELS_PATH)
    X = df.drop(label, axis=1)
    y = df[label]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42,
                                                        shuffle=True)

    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2,
                                                                    random_state=42,
                                                                    shuffle=True)

    X_train, y_train = preprocess(X_train, y_train)
    X_validation, _ = preprocess(X_validation)

    X_train = X_train.drop(columns=COLS_TO_DROP)


    cols = X_train.columns
    X_validation = X_validation.reindex(columns=cols, fill_value=0)
    X_train.fillna(0, inplace=True)
    X_validation.fillna(0, inplace=True)

    # pca plots
    #eda_plot_cancer_patient(X_train, y_train)



    model = TumorSizeRegressor()

    model.fit(X_train, y_train)
    train_mse = model.loss(X_train, y_train)
    validation_mse = model.loss(X_validation, y_validation)

    print('train mse: ', train_mse)
    print('validation mse: ', validation_mse)


def unsupervised():
    label = 'tumor_size'
    df = load_data(
        TRAIN_DATA_PATH,
        TRAIN_LABELS_PATH)
    X = df.drop(label, axis=1)
    y = df[label]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42,
                                                        shuffle=True)

    X_train, X_validation, y_train, y_validation = train_test_split(X_train,
                                                                    y_train,
                                                                    test_size=0.2,
                                                                    random_state=42,
                                                                    shuffle=True)

    X_train, y_train = preprocess(X_train, y_train)
    X_validation, _ = preprocess(X_validation)

    X_train = X_train.drop(columns=COLS_TO_DROP)

    cols = X_train.columns
    X_validation = X_validation.reindex(columns=cols, fill_value=0)
    X_train.fillna(0, inplace=True)
    X_validation.fillna(0, inplace=True)


    # pca plots
    eda_plot_cancer_patient_spec(X_train, y_train)
    #print(kmeans_cluster(X_train))
    #print(spectral_cluster(X_train))


if __name__ == '__main__':
    label = 'tumor_size'
    df = load_data(
        TRAIN_DATA_PATH,
        TRAIN_LABELS_PATH)
    X = df.drop(label, axis=1)
    y = df[label]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42,
                                                        shuffle=True)

    X_train, X_validation, y_train, y_validation = train_test_split(X_train,
                                                                    y_train,
                                                                    test_size=0.2,
                                                                    random_state=42,
                                                                    shuffle=True)

    X_train, y_train = preprocess(X_train, y_train)
    X,y=preprocess(X,y)
    X=X.drop(columns=COLS_TO_DROP)
    X.fillna(0, inplace=True)
    regularization_path(X,y,X.columns)
    #X_validation, _ = preprocess(X_validation)

    #X_train = X_train.drop(columns=COLS_TO_DROP)

    #cols = X_train.columns
    #X_validation = X_validation.reindex(columns=cols, fill_value=0)
    #X_train.fillna(0, inplace=True)
    #X_validation.fillna(0, inplace=True)
    #unsupervised()
    #print(summarize_dataframe(X_train))
    #cluster_samples(X_train)
    #pear_ridg_cor(X,y)

    cluster_metastasis_mark(X)
