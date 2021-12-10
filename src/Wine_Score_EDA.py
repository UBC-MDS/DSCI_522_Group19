#!/usr/bin/env python
# coding: utf-8

"""
Script for creating plots as part of exploratory data analysis

Usage: Wine_Score_EDA.py --input_file=<input_file> --out_dir=<out_dir>

Options: 
  
--input_file=<input_file> Path (including filename) to processed data with "train_df"
--out_dir=<out_dir>       Path to directory where EDA results should be saved

"""

# Example:
# python src/Wine_Score_EDA.py --input_file=<input_file> --out_dir="out_dir"



from docopt import docopt
import os

import pandas as pd
import altair as alt
from altair_saver import save
alt.data_transformers.enable('data_server')
alt.renderers.enable('mimetype')


opt = docopt(__doc__) 

def main(input_file, out_dir):
    # read train_df.csv file
    train_df = pd.read_csv(input_file)
    summary_of_missing_data(train_df, out_dir)
    figures(train_df, out_dir)

def summary_of_missing_data(train_df, out_dir):
    """
    Creates csv file with information about missing data in the dataset

    Parameters
    ----------
    input_file :
    train_df training data set 

    Returns
    ---------
    csv file with summary about missing data
    
    """
    na_df = train_df.isna().sum().reset_index()
    na_df.columns = ['feature', 'number_of_missing_observations']
    
    try:
        na_df.to_csv(f'{out_dir}/missing_observations_summary.csv', index=False)
    except:
        os.makedirs(out_dir, exist_ok=True)
        na_df.to_csv(f'{out_dir}/missing_observations_summary.csv', index=False)
    
 
def figures(train_df, out_dir):
    """
    Creates and saves charts as images in results folder

    Parameters
    ----------
    input_file :
    train_df training data set 

    Returns
    ---------
    png files:
    quality_fig 
    repeat_plots
    cor_plot   
    """
    #create quality figure distribution - quality_fig
    quality_fig = alt.Chart(train_df, title = "Distribution of quality scores").mark_bar().encode(
    x=alt.X('quality', bin=alt.Bin(maxbins=7)),
    y='count()',
    tooltip='count()')

    #create numeric feature distribution - repeat_plots
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
      )).properties(title="Data distribution of numeric features in training datasets")

    #create correlation figure - cor_plot
    cor_data = (
    train_df.corr()
    .stack()
    .reset_index()
    .rename(columns={0: "correlation", "level_0": "variable", "level_1": "variable2"})
    )

    cor_data["correlation_label"] = cor_data["correlation"].map(
      "{:.2f}".format
    )  

    base = alt.Chart(cor_data, title = "Quality distribution of wines in the training and test datasets").encode(x="variable2:O", y="variable:O")

    text = base.mark_text().encode(
      text="correlation_label",
      color=alt.condition(
          alt.datum.correlation > 0.5, alt.value("white"), alt.value("black")
      ),
    )

    cor_plot = base.mark_rect().encode(
      alt.Color("correlation:Q", scale=alt.Scale(domain=(-1, 1), scheme="purpleorange"))
    )

    cor_plot = (
      (cor_plot + text)
      .properties(height=600, width=600)
      .configure_axis(labelFontSize=16)
      .configure_legend(titleFontSize=15)
    )

    #saves all plots to results folder
    try:
        quality_fig.save(f'{out_dir}/quality_dist.png')
        repeat_plots.save(f'{out_dir}/repeat_plots.png')
        cor_plot.save(f'{out_dir}/cor_plot.png')
    except:
        os.makedirs(out_dir, exist_ok=True)
        quality_fig.save(f'{out_dir}/quality_dist.png')
        repeat_plots.save(f'{out_dir}/repeat_plots.png')
        cor_plot.save(f'{out_dir}/cor_plot.png')

if __name__ == "__main__":
  main(opt['--input_file'], opt['--out_dir'])
