import matplotlib.pyplot as plt
import seaborn as sns


class Visualization():

    def __init__(self, df) -> None:
        self.df = df
    
    def stats(self):
        """
        Data statistics and general information
        """

        # Check the shape of the dataset
        display("Data Shape:", self.df.shape)

        # Check the data types of the columns
        display("Data Types", self.df.dtypes)

        # Check for missing values
        display("Count of missing values", self.df.isnull().sum())

        # Check summary statistics of the numerical columns
        display("Overall Statistics", self.df.describe())

    def feature_distribution(self, feature_name):
        """
        To plot a feature distribution
        """

        # Visualize the distribution of a feature
        sns.histplot(self.df[feature_name], kde=False)
        plt.savefig('charts/eda/feature_distribution/{}.png'.format(feature_name))

    def feature_correlation_scatterplot(self, feature_name_x, feature_name_y):
        """
        To plot correlation of two numberical features
        """

        # Visualize the correlation of two numerical features
        sns.scatterplot(x=feature_name_x, y=feature_name_y, data=self.df)
        plt.savefig('charts/eda/correlation_distribution/{}_{}.png'.format(feature_name_x, feature_name_y))

    def feature_correlation_boxplot(self, feature_name_categorical, feature_name_numerical):
        """
        To plot correlation between a categorical and a numberical feature
        """

        # Visualize the correlation between a categorical and a numerical feature
        sns.boxplot(x=feature_name_categorical, y=feature_name_numerical, data=self.df, hue="label")
        plt.savefig('charts/eda/correlation_distribution/{}_{}.png'.format(feature_name_categorical, feature_name_numerical))


        
def display(message="Display Message:", df=None):
    """
    Display a message on a console
    """

    print(message)
    print(df)
    print("----------------------------------------")