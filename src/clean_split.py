"""Cleans, splits and pre-processes data
Writes the training and test data to separate feather files

Usage: clean_split.py --input_red=<input_red> --input_white=<input_white> --out_dir=<out_dir>

Options: 
--input_red=<input_red>           Path (including filename) to raw data with "red wine"
--input_white=<input_white>       Path (including filename) to raw data with "white wine"
--out_dir=<out_dir>  Path to directory where the processed data should be written

"""

# Example:
# python src/clean_split.py --input_red="data/raw/winequality-red.csv" --input_white="data/raw/winequality-white.csv" --out_dir="data/processed/"

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from docopt import docopt

opt = docopt(__doc__)  # parse these into dictionary opt


def main(input_red, input_white, out_dir):
    # input_red = "data/raw/winequality-red.csv"
    # input_white = "data/raw/winequality-white.csv"
    # out_file = "data/processed/train_df.csv"
    
    wine_df_original = combine_dataframes(input_red, input_white)
    train_df, test_df = train_test_split(wine_df_original, test_size=0.2, random_state=123)
    X_train, X_test, y_train, y_test = split_for_train_test(wine_df_original, target_column='quality', test_size=0.2, random_state=123)
    
    for df in [train_df, test_df, X_train, X_test, y_train, y_test]:
        for out_name in ['train_df', 'test_df', 'X_train', 'X_test', 'y_train', 'y_test']:
            df.to_csv(out_dir + out_name + '.csv', index=False)
    
    
def combine_dataframes(input_red, input_white):
    red_df = pd.read_csv(input_red, sep=";")
    white_df = pd.read_csv(input_white, sep=";")
    

    red_df['wine_type'] = 'red_wine'
    white_df['wine_type'] = 'white_wine'
    wine_df = pd.concat([red_df,white_df]).reset_index().drop(columns = ['index'])
    
    return wine_df

def split_for_train_test(original_df, target_column, test_size=0.2, random_state=123):

    train_df, test_df = train_test_split(original_df, test_size=test_size, random_state=random_state)
    X_train = train_df.drop(columns=[target_column])
    X_test = test_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    y_test = test_df[target_column]
    
    return X_train, X_test, y_train, y_test
    
    
    
if __name__ == "__main__":
    main(opt['--input_red'], opt['--input_white'], opt['--out_dir'])    