#!/usr/bin/env python
# coding: utf-8

"""Loads train dataset
creates figures as part of exploratory data analysis

Usage: Wine_Score_EDA.py --input_file=<train_df>

Options: 
  
--input_file=<train_df>           Path (including filename) to processed data with "train_df"

"""

# Example:
# python src/Wine_Score_EDA.py --input_file=<train_df>



from docopt import docopt

import pandas as pd
import altair as alt
from altair_saver import save
alt.data_transformers.enable('data_server')
alt.renderers.enable('mimetype')


opt = docopt(__doc__) 

def main(input_file):
  train_df = read_file(input_file)
  figures(train_df)

def read_file(input_file):
  train_df=pd.read_csv(input_file)
  return train_df


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
  main(opt['--input_file'])