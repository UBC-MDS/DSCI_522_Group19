"""Downloads data csv data from the web to a local filepath as a csv.

Usage: download_data.py --url=<url> --out_file=<out_file>

Options: 
--url=<url>    URL from where to download the data (must standard csv format)
--out_file=<out_file>    Path (including the filename) of where to write the file with downloaded data locally

"""

# Example:
# python download_data.py --url="https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv" --out_file="../data/raw/winequality-red.csv"
# python download_data.py --url="https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv" --out_file="../data/raw/winequality-white.csv"

import os
import pandas as pd
import requests
from docopt import docopt

opt = docopt(__doc__)  # parse these into dictionary opt

def main(url, out_file):
    """Take the url, download the data from the url and save as .csv file.

    Parameters:
    url (str): the raw url of the data set in a csv format 
    out_file (str):  Path (including the filename) of where to write the file with downloaded data locally

    Returns:

    Example:
    main("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", "data/red_wine.csv")

    """
    # red wine url: url_file= "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    # white wine url: url_file = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
    # red wine output file: out_red = "../data/raw/winequality-red.csv"
    # white wine output file: out_white = "../data/raw/winequality-white.csv"
    try: 
        request = requests.get(url)
        request.status_code == 200
    except Exception as req:
        print(req)
        print("Website at the provided url does not exist")
        
    
    data = pd.read_csv(url, sep=";")
    
    try:
        data.to_csv(out_file, index=False)
    except:
        os.makedirs(os.path.dirname(out_file))
        data.to_csv(out_file, index=False)


if __name__ == "__main__":
    main(opt['--url'], opt['--out_file'])