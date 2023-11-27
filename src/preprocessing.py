import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class Preprocessing():

    def __init__(self, df) -> None:
        self.df = df
        self.target_column = "label"

    def run(self) -> None:
        """
        Start cleaning the data
        """
        
        # Replace values
        self.replace_values("imd_band", "20-Oct", "10-20%")
        self.replace_values("code_presentation", ["2013B","2013J","2014B","2014J"], ["2013-Feb","2013-Oct","2014-Feb","2014-Oct"])
        
        # Fill missing values
        self.fill_values("imd_band")
        
        # Create twenty-two features
        self.feature_extraction()
        
        # Drop useless features
        self.drop_features()

        # Save clean dataframe in a new file
        self.save_new_dataset()
        
        # Display statistics of dataset
        # self.visualization.stats()
        
        
        
    def replace_values(self, column, old, new):
        """
        To replace some values with new ones
        """
        self.df[column].replace(old, new, inplace=True)
        
    
    def fill_values(self, column):
        """
        Filling missing values of columns with 'mode' of values
        """
        self.df[column] = self.df.apply(lambda row: self.df[self.df['region']==row['region']]['imd_band'].mode()[0] if pd.isna(row['imd_band']) else row['imd_band'], axis=1)
        
    
    def feature_extraction(self):
        
        """
        add twenty-two features to the dataframe in order to model effectively
        """
        step = 20 # number of activities
        total_columns = len(self.df.columns) - 1
        # insert a column, sum of clicks of specific activity
        insert = lambda i: self.df.iloc[:, i:total_columns:step].sum(axis=1)
        new_columns = {"total_dataplus_clicks":insert(9), "total_dualpane_clicks":insert(10),
              "total_external_quiz_clicks":insert(11), "total_folder_clicks":insert(12),
              "total_forum_clicks":insert(13), "total_glossary_clicks":insert(14),
              "total_homepage_clicks":insert(15), "total_hrml_clicks":insert(16),
              "total_collaborate_clicks":insert(17), "total_content_clicks":insert(18),
              "total_elluminate_clicks":insert(19), "total_wiki_clicks":insert(20),
              "total_page_clicks":insert(21), "total_questionnaire_clicks":insert(22),
              "total_quiz_clicks":insert(23), "total_repeated_clicks":insert(24),
              "total_resource_clicks":insert(25), "total_shared_clicks":insert(26),
              "total_subpage_clicks":insert(27), "total_url_clicks":insert(28)}
              
        for col,values in new_columns.items():
    
            self.df.insert(loc=len(self.df.columns)-1,
            column=col,
            value=values)
        
    
        # sum up the total activities each student participate
        def total_activity(row):
            """
            """
            total = 0
            for r in row:
                if r != 0:
                    total += 1
            
            return total
        
        
        total_activities = self.df.iloc[:,9:-21].apply(total_activity, axis=1)
        self.df.insert(loc = len(self.df.columns)-1,
                       column = "total_activities",
                       value = total_activities)
        
        # sum up the total clicks for each student
        total_clicks = self.df.iloc[:,9:-22].sum(axis=1)
        self.df.insert(loc = len(self.df.columns)-1,
                      column = "total_clicks",
                      value = total_clicks)
        
        
    def drop_features(self):
        """
        """
        self.df.drop(self.df.columns[9:-23], axis=1, inplace=True)
        

    def save_new_dataset(self):
        """
        To save cleaned dataframe as a new csv file
        """
        

        # Save the updated df
        self.df.to_csv("datasets/clean_ds.csv", index=False)
        

# Feature Encoding
def feature_encoding(df, columns):
    """
    Encode categorical features
    """
    
    # Remove id_student column
    df.drop("id_student", axis=1, inplace=True)
    
    # Remove outliers
    df.drop([11360, 8173, 9449, 14977, 20177, 2333, 1950, 18371, 19110], inplace=True)
    
    # To Encode categorical features
    le = LabelEncoder()
    for col in columns:
        df[col] = le.fit_transform(df[col])
        
        
        
    return df
    
    