"""Transforms train data and creates model fitting with Ridge/SVC/OneVsRest and RandomForest algorithms

Usage: src/model_fitting.py --X_train_path=<X_train_path>  --X_test_path=<X_test_path> --y_train_path=<y_train_path> --y_test_path=<y_test_path>

Options: 
--X_train_path=<X_train_path>     Path (including filename) data with Xtrain split
--X_test_path=<X_test_path>       Path (including filename) data with Xtest split
--y_train_path=<y_train_path>     Path (including filename) data with ytrain split
--y_test_path=<y_test_path>       Path (including filename) data with ytest split

"""

# Example:
# python src/model_fitting.py --X_train_path="data/processed/X_train.csv" --X_test_path="data/processed/X_test.csv" --y_train_path="data/processed/y_train.csv" --y_test_path="data/processed/y_test.csv"

import pandas as pd
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.svm import SVC, SVR

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    average_precision_score, 
    auc
)

from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
    cross_validate,
    train_test_split,
)

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel

from docopt import docopt

opt = docopt(__doc__)  # parse these into dictionary opt


def main(X_train_path, X_test_path, y_train_path, y_test_path):
    """
    Reads data from two input files and combines data into one dataframe

    Parameters:
    ----------
    input_red : path for the first input file 
    input_white : path for the second input file 
    out_dir: directory  where output files are saved

    Returns:
    --------
    train_df, test_df : creates csv files for train_df, test_df
    X_train, X_test, y_train, y_test : creates csv files for X_train, X_test, y_train, y_test

    example:
    --------
        input_red = "data/raw/winequality-red.csv"
        input_white = "data/raw/winequality-white.csv"
        out_file = "data/processed/train_df.csv"
    """

    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)

    y_train = pd.read_csv(y_train_path)
    y_test = pd.read_csv(y_test_path)
    
    transform_with_pipe(X_train, y_train)


def transform_with_pipe(X_train, y_train):
    numeric_feats = X_train.select_dtypes(include=[np.number]).columns.values.tolist()
    binary_feats = ["wine_type"]

    numeric_transformer = make_pipeline(StandardScaler())
    binary_transformer = make_pipeline(OneHotEncoder(drop="if_binary", dtype=int))

    preprocessor = make_column_transformer(
        (numeric_transformer, numeric_feats),
        (binary_transformer, binary_feats)
    )

    column_names = numeric_feats + binary_feats
    train_df_transformed = pd.DataFrame(preprocessor.fit_transform(X_train, y_train), columns = column_names)
    train_df_transformed.to_csv("data/processed/Xtrain_transformed.csv", index = False)


if __name__ == "__main__":
    main(opt['--X_train_path'], opt['--X_test_path'], opt['--y_train_path'], opt['--y_test_path'])    