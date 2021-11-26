#!/usr/bin/env python
# coding: utf-8

"""
Script for creating plots as part of exploratory data analysis

Usage: Wine_Score_EDA.py --input_file=<input_file>

Options: 
  
--input_file=<input_file> Path (including filename) to processed data with "train_df"

"""

# Example:
# python src/Wine_Score_EDA.py --input_file=<input_file>



from docopt import docopt

import pandas as pd
import altair as alt
from altair_saver import save
alt.data_transformers.enable('data_server')
alt.renderers.enable('mimetype')


opt = docopt(__doc__) 

def main(input_file):
  # read train_df.csv file
  train_df=pd.read_csv(input_file)
  figures(train_df)



def figures(train_df):
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
  #create quality figure distribution
  quality_fig = alt.Chart(train_df).mark_bar().encode(
    x=alt.X('quality', bin=alt.Bin(maxbins=7)),
    y='count()',
    tooltip='count()')
  
  #save quality_dist plot to results folder 
  save(quality_fig, "results/quality_dist.png")
  
  #create numeric feature distribution
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

  #save feature count distribution plot to results folder
  save(repeat_plots, "results/repeat_plots.png")
  
  cor_data = (
    train_df.corr()
    .stack()
    .reset_index()
    .rename(columns={0: "correlation", "level_0": "variable", "level_1": "variable2"})
    )

  cor_data["correlation_label"] = cor_data["correlation"].map(
      "{:.2f}".format
  )  
  
  base = alt.Chart(cor_data).encode(x="variable2:O", y="variable:O")
  
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

#save correlation plot to results folder
  save(cor_plot, "results/cor_plot.png")

  return quality_fig, repeat_plots, cor_plot
    
if __name__ == "__main__":
  main(opt['--input_file'])
