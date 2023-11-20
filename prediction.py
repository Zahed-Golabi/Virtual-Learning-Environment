import numpy as np
import pandas as pd
from src.preprocessing import Preprocessing



if __name__ == "__main__":
    
    # Load the dataset into a Pandas DataFrame
    df = pd.read_csv('./datasets/Clickstream-MOOC-dataset 1.csv')

    # Start cleaning the dataset
    preprocessing = Preprocessing(df)
    preprocessing.run()
    
    # Load the clean dataset into a Pandas DataFrame
    # df = pd.read_csv('./datasets/clean_ds.csv')


