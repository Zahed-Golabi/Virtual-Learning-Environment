import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
from src.visualization import display
import pickle
import sys

# models
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# sampling
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


import warnings

warnings.filterwarnings("ignore")


class Training:
    def __init__(self, df, task="binary") -> None:
        self.df = df
        self.task = task
        self.models = {}
        self.X = df.drop("label", axis=1)
        self.y = df["label"]

    def fit_binary(self):
        """
        Fit all models on our dataset
        """

        # split data into train, test
        self.df["label"] = self.df["label"].apply(
            lambda x: "Failure" if (x == "Withdrawn" or x == "Fail") else "Success"
        )
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.30, random_state=30
        )

        # define models
        self.models["Neural Network"] = MLPClassifier(
            solver="lbfgs",
            alpha=1e-5,
            hidden_layer_sizes=(5, 10, 10, 2),
            random_state=1,
        )
        self.models["Gradient Boosting"] = GradientBoostingClassifier()
        self.models["Logistic Regression"] = LogisticRegression()
        self.models["Support Vector Machine"] = LinearSVC()
        self.models["Decision Tree"] = DecisionTreeClassifier()
        self.models["Random Forest"] = RandomForestClassifier(
            bootstrap=True,
            max_depth=42,
            max_features="auto",
            min_samples_leaf=2,
            min_samples_split=90,
            n_estimators=420,
        )
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
            self.models[key].fit(X_train, y_train)
            # make predictions
            predictions = self.models[key].predict(X_test)

            # plot and save confusion matrix
            print("============================================================")

            # display model metrics
            print(key)
            cm = confusion_matrix(y_test, predictions, labels=["Success", "Failure"])
            cmd = ConfusionMatrixDisplay(cm, display_labels=["Success", "Failure"])
            cmd.plot()
            plt.savefig(f"./models/binary/{key}_cm.jpeg")
            plt.close()
            print()
            print(cm)
            print(classification_report(y_test, predictions))
            print("============================================================\n")

            # calculate metrics
            accuracy[key] = accuracy_score(
                predictions,
                y_test,
            )
            precision[key] = precision_score(predictions, y_test, pos_label="Success")
            recall[key] = recall_score(predictions, y_test, pos_label="Success")

        # save metrics
        total_metrics = pd.DataFrame(
            index=self.models.keys(), columns=["Accuracy", "Precision", "Recall"]
        )
        total_metrics["Accuracy"] = accuracy.values()
        total_metrics["Precision"] = precision.values()
        total_metrics["Recall"] = recall.values()

        # sort metric values based on Accuracy
        total_metrics.sort_values(by="Accuracy", ascending=False, inplace=True)

        # save total_metrics
        total_metrics.to_csv("models/binary/metrics.csv")

    def fit_multiclass(self, kind="smote", imbalanced=True):

        if imbalanced == False:

            X, y = self.X, self.y

        elif kind == "near_miss":

            nm = NearMiss()
            X, y = nm.fit_resample(self.X, self.y)

        elif kind == "smote":

            smote = SMOTE()
            X, y = smote.fit_resample(self.X, self.y)

        elif kind == "random_oversampling":

            ros = RandomOverSampler(random_state=42)
            X, y = ros.fit_resample(self.X, self.y)

        elif kind == "random_undersampling":

            rus = RandomUnderSampler(random_state=42, replacement=True)
            X, y = rus.fit_resample(self.X, self.y)

        else:
            print("Kind is Invalid!")
            return "Error"

        # RandomForest Classifier
        rf = RandomForestClassifier(
            bootstrap=True,
            max_depth=70,
            max_features="auto",
            min_samples_leaf=2,
            min_samples_split=85,
            n_estimators=420,
        )

        # Train, Test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, random_state=13
        )

        rf.fit(X_train, y_train)

        # Make predictions
        predictions = rf.predict(X_test)

        # Calculate metrics
        cm = confusion_matrix(y_test, predictions)

        print(
            "\n========================================================================"
        )
        if imbalanced:
            print(f"\nAlgorithm to sampling: {kind}\n")
        else:
            print("\nSupposing data is balanced\n")

        print(cm)
        print(classification_report(y_test, predictions))
        print(
            "\n========================================================================"
        )
