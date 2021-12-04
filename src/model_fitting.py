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

from scipy.stats import randint

import imblearn
from imblearn.pipeline import make_pipeline as make_imb_pipeline
from imblearn.over_sampling import RandomOverSampler

from docopt import docopt

opt = docopt(__doc__)  # parse these into dictionary opt


def main(X_train_path, X_test_path, y_train_path, y_test_path):
    """
    Transforms train data and performs evaluation of multiple models with selecting one model with the best scores

    Parameters:
    ----------
    X_train_path : path for csv file with X_train data
    X_test_path : path for csv file with X_test data
    y_train_path: path for csv file with y_train data
    y_test_path : path for csv file with y_test data

    Returns:
    --------
    csv files with results of cross-validation and parameters of the best model store in:
        'results/cv_scores_for_alternative_methods.csv'
        'results/final_results.csv'

    """

    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)

    y_train = pd.read_csv(y_train_path)
    y_test = pd.read_csv(y_test_path)
    
    # Limiting amount of data to 100 rows only. Used for debugging or making any modifications:
    # X_train = X_train.head(100)
    # X_test = X_test.head(100)
    # y_train = y_train.head(100)
    # y_test = y_test.head(100)
    
    preprocessor = transform_with_pipe(X_train, y_train)
    evalute_alternative_methods(X_train, y_train, preprocessor)
    tune_hyperparameters_for_best_model(X_train, X_test, y_train, y_test, preprocessor)
 

def transform_with_pipe(X_train, y_train):
    """
    Transforms columns for train dataframe
    
    Parameters
    ----------
    X_train : numpy array or pandas DataFrame
        X in the training data
    y_train : numpy array or pandas DataFrame
        y in the training data

    Returns
    ----------
        preprocessor object from pipe of column transformers
    """
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
    
    return preprocessor


def mean_std_cross_val_scores(model, X_train, y_train, **kwargs):
    """
    Returns mean and std of cross validation
    (Taken from UBC DSCI 573 Lecture Notes)
    
    Parameters
    ----------
    model :
        scikit-learn model
    X_train : numpy array or pandas DataFrame
        X in the training data
    y_train :
        y in the training data

    Returns
    ----------
        pandas Series with mean scores from cross_validation
    """

    scores = cross_validate(model, X_train, y_train, **kwargs)

    mean_scores = pd.DataFrame(scores).mean()
    std_scores = pd.DataFrame(scores).std()
    out_col = []

    for i in range(len(mean_scores)):
        out_col.append((f"%0.3f (+/- %0.3f)" % (mean_scores[i], std_scores[i])))

    return pd.Series(data=out_col, index=mean_scores.index)

def mape(true, pred):
    """
    Calculates Mean Absolute Percentage Error
    (Taken from UBC DSCI 573 Lecture Notes)
    
    Parameters
    ----------
    true : numpy array with actual values

    pred : numpy array with predicted values

    Returns
    ----------
        numerical value with calculated MAPE
    """
        
    return 100.0 * np.mean(np.abs((pred - true) / true))

def evalute_alternative_methods(X_train, y_train, preprocessor):
    """
    Performes evaluation of relevant models with screening based on the highest cross-validataion score
    
    Parameters
    ----------
    X_train : numpy array or pandas DataFrame
        X in the training data
    y_train :
        y in the training data
    preprocessor: 
        preprocessor object from pipe of column transformers

    Returns
    ----------
        writes results to csv file in 'results/cv_scores_for_alternative_methods.csv'
    """
    mape_scorer = make_scorer(mape, greater_is_better=False)
    
    score_types_reg = {
        "neg_mean_squared_error": "neg_mean_squared_error",
        "neg_root_mean_squared_error": "neg_root_mean_squared_error",
        "neg_mean_absolute_error": "neg_mean_absolute_error",
        "r2": "r2",
    }

    models = {
        "Ridge": Ridge(max_iter=50),
        "SVC": SVC(),
        "OneVsRest":OneVsRestClassifier(LogisticRegression()),
        "Random Forest": RandomForestRegressor(random_state=123)
    }
    
    results_comb={}
    for keys in models.keys():
        pipe_comb = make_imb_pipeline(RandomOverSampler(sampling_strategy="minority"), preprocessor, models[keys])
        results_comb[keys]=mean_std_cross_val_scores(
            pipe_comb, X_train, y_train, return_train_score=True, scoring=score_types_reg
    )
    
    
    """
    After comparing different regression models by using various matrix, we found the better model is Random Forest, because we got highest cross-validation score. 
    However, we figured out that we may encounter some overfitting issue with Random Forest model as the difference between train score and validation score is quite wide. 
    So, we further conduct feature selections and hyper-parameter optimization as follow:
    """
    rfe = RFE(RandomForestRegressor(random_state=123), n_features_to_select=10)
    pipe_rf_rfe = make_imb_pipeline(RandomOverSampler(sampling_strategy="minority"), preprocessor, rfe, RandomForestRegressor(random_state=123))
    results_comb['Random Forest_rfe'] = mean_std_cross_val_scores(pipe_rf_rfe, X_train, y_train, return_train_score=True, scoring=score_types_reg)


    results_df = pd.DataFrame(results_comb)
    results_df.to_csv('results/cv_scores_for_alternative_methods.csv')
    
def tune_hyperparameters_for_best_model(X_train, X_test, y_train, y_test, preprocessor):
    """
    Uses RandomSearchCV for hyperparameter tuning of the best model (RandomForestRegressor)
    
    Parameters
    ----------
    X_train : numpy array or pandas DataFrame
        X in the training data
    X_test: numpy array or pandas DataFrame
        X in the test data
    y_train :
        y in the training data
    y_test: numpy array or pandas DataFrame
        y in the test data
    preprocessor: 
        preprocessor object from pipe of column transformers

    Returns
    ----------
        writes results to csv file in 'results/final_results.csv'
    """
    rfe = RFE(RandomForestRegressor(random_state=123), n_features_to_select=10)
    pipe_rf_rfe = make_pipeline(preprocessor, rfe, RandomForestRegressor(random_state=123))
    
    param_dist = {"randomforestregressor__max_depth": randint(low=5, high=1000),
                 "randomforestregressor__max_leaf_nodes": randint(low=5, high=1000),
                 "randomforestregressor__n_estimators": randint(low=5, high=1000),}

    random_search = RandomizedSearchCV(
        pipe_rf_rfe,
        param_distributions=param_dist,
        n_jobs=-1,
        n_iter=10,
        cv=5,
        random_state=123
    )

    random_search.fit(X_train, y_train.values.ravel())   

    cv_best_score = random_search.best_score_
    train_score = random_search.score(X_train, y_train)
    test_score = random_search.score(X_test, y_test)

    final_results_dict = {'best_model': 'RandomForestRegressor', 
                          'max_depth': random_search.best_params_['randomforestregressor__max_depth'], 
                          'max_leaf_nodes': random_search.best_params_['randomforestregressor__max_leaf_nodes'], 
                          'n_estimators': random_search.best_params_['randomforestregressor__n_estimators'], 
                         'cv_best_score': round(cv_best_score, 3), 
                         'train_score': round(train_score, 3), 
                         'test_score': round(test_score, 3)}

    final_results_df = pd.DataFrame.from_dict(final_results_dict, orient='index')
    final_results_df.to_csv('results/final_results.csv', header=None)

if __name__ == "__main__":
    main(opt['--X_train_path'], opt['--X_test_path'], opt['--y_train_path'], opt['--y_test_path'])    