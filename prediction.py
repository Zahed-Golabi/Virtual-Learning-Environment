import numpy as np
import pandas as pd
from src.preprocessing import Preprocessing
from src.visualization import Visualization
from src.visualization import display




if __name__ == "__main__":
    
    # Load the dataset into a Pandas DataFrame
    # df = pd.read_csv('./datasets/Clickstream-MOOC-dataset 1.csv')

    # Start cleaning the dataset
    # preprocessing = Preprocessing(df)
    # preprocessing.run()
    
    # Load the clean dataset into a Pandas DataFrame
    # code_module, code_presentation, gender, region, highest_education, imd_band, age_band, disability, and total clicks for each activity
    df = pd.read_csv('./datasets/clean_ds.csv')
    visualization = Visualization(df)
    visualization.stats()
    visualization.distribution_plot("code_module", mode="bar")
    #categorical_features = ["code_module", "code_presentation", "gender", "region", "highest_education", "imd_band", "age_band", "disability"]
    #for feature in categorical_features:
    #    visualization.feature_distribution(feature)


