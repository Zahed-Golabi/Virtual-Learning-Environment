import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
from src.visualization import display
import pickle

# models
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

import warnings
warnings.filterwarnings("ignore")


class Training():

    def __init__(self, df, task="binary") -> None:
        self.df = df
        self.task = task
        self.models = {}
        self.X = df.drop("label", axis=1)
        self.y = df["label"]
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data()
        
    def split_data(self):
        """
        Split dataset to train and test
        """
        
        if self.task == "binary":
            self.df["label"] = self.df["label"].apply(lambda x: "Failure" if (x=="Withdrawn" or x=="Fail") else "Success")
        else:
            pass # Multiclass classification

        # Split data into training and testing sets
        return train_test_split(self.df.drop('label', axis=1), self.df['label'], test_size=0.3, random_state=30)
    
    def fit(self):
        """
        Fit all models on our dataset
        """
        self.models["Neural Network"] = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 10, 10, 2), random_state=1)
        self.models["Gradient Boosting"] = GradientBoostingClassifier()
        self.models["Logistic Regression"] = LogisticRegression()
        self.models["Support Vector Machine"] = LinearSVC()
        self.models["Decision Tree"] = DecisionTreeClassifier()
        self.models["Random Forest"] = RandomForestClassifier(bootstrap = True, max_depth = 42, max_features = 'auto', min_samples_leaf = 2, min_samples_split = 90, n_estimators = 420)
        self.models["Naive Bayes"] = GaussianNB()
        self.models["K-Nearest Neighbor"] = KNeighborsClassifier(n_neighbors=9)
        
        # metrics
        accuracy, precision, recall = {}, {}, {}
        
        for key in self.models.keys():
        
            # validate RandomForestClassifier using K-fold cross validation 
            # k = 10
            # kf = KFold(n_splits=k, shuffle=True, random_state=23)
            # random_forest = bootstrap=True, max_depth=42, max_features="auto", min_samples_leaf=2, min_samples_split=90, n_estimators=420)
            # result = cross_val_score(random_forest, self.X, self.y, cv=kf)
            # array([0.87919233, 0.87303217, 0.88056126, 0.88603696, 0.88193018,
                   # 0.88124572, 0.88740589, 0.87748118, 0.87709688, 0.87196166])
            
            
            # fit the classifier
            self.models[key].fit(self.X_train, self.y_train)
            
            # make predictions
            predictions = self.models[key].predict(self.X_test)
            
            # calculate metrics
            accuracy[key] = accuracy_score(predictions, self.y_test, )
            precision[key] = precision_score(predictions, self.y_test, pos_label="Success")
            recall[key] = recall_score(predictions, self.y_test, pos_label="Success")
            
        # save metrics
        total_metrics = pd.DataFrame(index=self.models.keys(), columns=["Accuracy","Precision","Recall"])
        total_metrics["Accuracy"] = accuracy.values()
        total_metrics["Precision"] = precision.values()
        total_metrics["Recall"] = recall.values()
        
        # sort metric values based on Accuracy
        total_metrics.sort_values(by="Accuracy", ascending=False, inplace=True)
        
        # save total_metrics
        total_metrics.to_csv("models/binary/metrics.csv")