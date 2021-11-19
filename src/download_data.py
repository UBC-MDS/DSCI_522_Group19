"""Downloads data csv data from the web to a local filepath as a csv.

Usage: download_data.py --url=<url> --out_file=<out_file>

Options: 
--url=<url>            URL from where to download the data (must standard csv format)
--out_file=<out_file>  Path (including the filename) of where to write the file locally

"""

import os
import pandas as pd
import requests
from docopt import docopt

opt = docopt(__doc__)  # parse these into dictionary opt

def main(url, out_file):
    # url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    # url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
    # out_file = "data/winequality-red.csv"
    # out_file = "data/winequality-white.csv"
    try: 
        request = requests.get(url)
        request.status_code == 200
    except Exception as req:
        print("Website at the provided url does not exist")
        
    
    data = pd.read_csv(url, header=None)
    
    try:
        data.to_csv(out_file, index=False)
    except:
        os.makedirs(os.path.dirname(out_file))
        data.to_csv(out_file, index=False)


if __name__ == "__main__":
    main(opt['--url'], opt['--out_file'])