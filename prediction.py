import numpy as np
import pandas as pd
from src.preprocessing import Preprocessing
from src.preprocessing import feature_encoding
from src.visualization import Visualization
from src.visualization import display
from src.visualization import feature_importance_plot
from src.visualization import model_evaluation_plot
from src.training import Training





if __name__ == "__main__":
    
    # Load the dataset into a Pandas DataFrame
    #df = pd.read_csv('./datasets/Clickstream-MOOC-dataset 1.csv')

    # Start cleaning the dataset
    #preprocessing = Preprocessing(df)
    #preprocessing.run()
    
    # Load the clean dataset into a Pandas DataFrame
    df = pd.read_csv('./datasets/clean_ds.csv')
    visualization = Visualization(df)
    visualization.stats()
    visualization.feature_distribution_plot("label", kind="bar")
    visualization.feature_correlation_barplot("code_module")
    visualization.feature_correlation_scatterplot("total_shared_clicks","total_clicks")
    visualization.feature_correlation_catplot("code_presentation", "total_activities", kind="violin")
    
    # Encoding features
    colunas = ["code_module","code_presentation","gender","region",
                "highest_education","imd_band","age_band","disability"]
    df = feature_encoding(df, colunas)
    df.to_csv("./datasets/encoded_ds.csv", index=False)
    
    
    # Binary Classification
    df = pd.read_csv("./datasets/encoded_ds.csv")
    feature_importance_plot(df, "RandomForest")
    
    # final features
    final_features = ["total_activities","total_clicks","total_content_clicks","code_module","total_shared_clicks",
                      "total_hrml_clicks","total_url_clicks","total_page_clicks","total_dualpane_clicks",
                      "total_external_quiz_clicks", "label"]
    #                  
    df = df[final_features]
    model = Training(df, "binary")
    model.fit_binary()
    
    # load metrics_df
    metrics = pd.read_csv("./models/binary/metrics.csv", index_col=0)
    model_evaluation_plot(metrics, "Accuracy")
    
    # MultiClass Classification
    df = pd.read_csv("./datasets/encoded_ds.csv")
    
    # final features
    final_features = ["total_activities","total_clicks","total_content_clicks","code_module","total_shared_clicks",
                     "total_hrml_clicks","total_url_clicks","total_page_clicks","total_dualpane_clicks",
                     "total_external_quiz_clicks", "label"]                  
    df = df[final_features]
    
    model = Training(df, "multiclass")
    model.fit_multiclass(kind="smote", imbalanced=True)
    
    


