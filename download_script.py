import os
import pandas as pd

def download_data(url, data_folder):
    data = pd.read_csv(url, header=None)
    if not os.path.exists(os.path.join(os.getcwd(), data_folder)):
        os.mkdir(data_folder)
    
    csv_name = url.split('/')[-1]
    os.chdir(data_folder)
    data.to_csv(csv_name, index=False)
    os.chdir('../')
    