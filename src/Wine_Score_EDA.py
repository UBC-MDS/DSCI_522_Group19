#!/usr/bin/env python
# coding: utf-8

"""Load, clean and split data into train and test data
creates figures as part of exploratory data analysis

Usage: Wine_Score_EDA.py --input_red=<input_red> --input_white=<input_white>

Options: 
  
--input_red=<input_red>           Path (including filename) to raw data with "red wine"
--input_white=<input_white>       Path (including filename) to raw data with "white wine"

"""

# Example:
# python src/Wine_Score_EDA.py --input_red=data/winequality-red.csv --input_white=data/winequality-white.csv



from docopt import docopt

from sklearn.model_selection import train_test_split

import pandas as pd
import altair as alt
from altair_saver import save
alt.data_transformers.enable('data_server')
alt.renderers.enable('mimetype')


opt = docopt(__doc__) 

def main(input_red, input_white):
  wine_df_original = combine_dataframes(input_red, input_white)
  train_df, test_df = train_test_split(wine_df_original, test_size=0.2, random_state=123)
  figures(train_df)


def combine_dataframes(input_red, input_white):
  red_df = pd.read_csv(input_red, sep=";")
  white_df = pd.read_csv(input_white, sep=";")
  red_df['wine_type'] = 'red_wine'
  white_df['wine_type'] = 'white_wine'
  wine_df = pd.concat([red_df,white_df]).reset_index().drop(columns = ['index'])
  return wine_df


def figures(train_df):
  quality_fig = alt.Chart(train_df).mark_bar().encode(
    x=alt.X('quality', bin=alt.Bin(maxbins=7)),
    y='count()',
    tooltip='count()')
    
  save(quality_fig, "results/quality_dist.png")

  repeat_plots = (alt.Chart(train_df).mark_bar().encode(
    alt.X(alt.repeat(), type="quantitative", bin=alt.Bin(maxbins=40)),
    y="count()",
    color="wine_type",
    ).properties(width=200, height=100).repeat((
      train_df.select_dtypes(include=["int", "float"])
      .drop(["quality"], axis=1)
      .columns.to_list()
      ),
      columns=3
      ))
  save(repeat_plots, "results/repeat_plots.png")
  return quality_fig, repeat_plots
    
if __name__ == "__main__":
  main(opt['--input_red'], opt['--input_white'])