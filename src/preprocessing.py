import numpy as np
import pandas as pd
import os


class Preprocessing():

    def __init__(self, df) -> None:
        self.df = df
        self.target_column = "label"

    def run(self) -> None:
        """
        Start cleaning the data
        """
        
        
        # Create two features
        self.feature_engineering()
        
        # Drop useless features
        self.drop_features()

        # Save clean dataframe in a new file
        self.save_new_dataset()
        
        # Display statistics of dataset
        # self.visualization.stats()
        
        
    def feature_engineering(self):
        
        """
        add two features to the dataframe in order to model effectively
        """
    
        # sum up the total activities each student participate
        def total_activity(row):
            """
            """
            total = 0
            for r in row:
                if r != 0:
                    total += 1
            
            return total
        
        
        total_activities = self.df.iloc[:,9:-1].apply(total_activity, axis=1)
        self.df.insert(loc = len(self.df.columns)-1,
                       column = "total_activities",
                       value = total_activities)
        
        # sum up the total clicks for each student
        total_clicks = self.df.iloc[:,9:-2].sum(axis=1)
        self.df.insert(loc = len(self.df.columns)-1,
                      column = "total_clicks",
                      value = total_clicks)
        
        
    def drop_features(self):
        """
        """
        self.df.drop(self.df.columns[9:-3], axis=1, inplace=True)
        

    def save_new_dataset(self):
        """
        To save cleaned dataframe as a new csv file
        """
        

        # Save the updated df
        self.df.to_csv("datasets/clean_ds.csv", index=False)