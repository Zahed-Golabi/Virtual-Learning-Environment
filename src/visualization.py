import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px




class Visualization():

    def __init__(self, df) -> None:
        self.df = df
        self.target_column = "label"
        self.convert = {"code_module":"Modules", "code_presentation":"Semesters",
                        "gender":"Gender", "region":"Region", "highest_education":"Degree",
                        "imd_band":"Imd_band", "age_band":"Age_band", "disability":"Disability", "label":"Result"}
        
    
    def stats(self):
        """
        Data statistics and general information
        """

        # Check the shape of the dataset
        display("Data Shape:", self.df.shape)
        
        # Check the columns of the dataset
        display("columns:", self.df.columns)
        
        # Check the information of the dataset
        display("info:", self.df.info())
        
        # Check the distribution of the target feature
        display("targe:", self.df["label"].value_counts())

        # Check for missing values
        display("Count of missing values", self.df.isnull().sum())

        # Check summary statistics of the numerical columns
        display("Overall Statistics", self.df.describe())


    def feature_distribution_plot(self, feature_name_x, kind="bar"):
        """
        plot distribution for each feature
        """
        
        data = self.df.groupby(feature_name_x).agg({"id_student":"count"})
        data = data.reset_index().rename(columns={"id_student":"Count"})
        data["Percentage"] = data.apply(lambda row: round(100 *(row["Count"]/data["Count"].sum()),2), axis=1)
    
        if kind=="bar":
            fig = px.bar(data, x=feature_name_x, y="Count", text="Percentage", color=feature_name_x,
                        hover_data=["Count"], template="seaborn")
            fig.update_layout(margin=dict(l=20, r=50, t=50, b=50),
                         title="Distribution of: " + feature_name_x,
                         xaxis_title=self.convert[feature_name_x],
                         yaxis_title="Count",
                         legend_title=self.convert[feature_name_x],
                         width=600,
                         height=400,
                         uniformtext_minsize=10,
                         uniformtext_mode="hide")
    
        elif kind=="pie":
            fig = px.pie(data, values="Percentage", names=feature_name_x, title="Distribution of: " + feature_name_x)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            
        else:
            print("Invalid kind: \n kind is 'bar' or 'pie' ")
            exit()
        
    
        # save the figure
        fig.write_image(f"charts/eda/feature_distribution/{feature_name_x}_{kind}.jpeg")
        
    
    def feature_correlation_barplot(self, feature_name_x):
        """
        Plot distribution for feature and target
        """
        
        data = self.df.groupby([feature_name_x,self.target_column]).agg({"id_student":"count"})
        data = data.reset_index().rename(columns={"id_student":"Count"})
        data["Percentage"] = data.apply(lambda row: round(100 *(row["Count"]/data[data[feature_name_x]==row[feature_name_x]]["Count"].sum()),2), axis=1)
    

        fig = px.bar(data, x=feature_name_x, y="Percentage", text="Percentage", color=self.target_column,
                        hover_data=["Percentage"], barmode="stack", template="seaborn")
        fig.update_layout(margin=dict(l=20, r=50, t=50, b=20),
                         title=self.convert[feature_name_x] + " Correlation with Result",
                         xaxis_title=self.convert[feature_name_x],
                         yaxis_title="Percentage",
                         legend_title=self.convert[self.target_column],
                         width=800,
                         height=500,
                         uniformtext_minsize=10,
                         uniformtext_mode="hide")
    
        # save the figure
        fig.write_image(f"charts/eda/correlation_distribution_bar/{feature_name_x}.jpeg")
    
    
    def feature_correlation_scatterplot(self, feature_name_x, feature_name_y):
        """
        To plot correlation of two numberical features
        """

        # Visualize the correlation of two numerical features
        fig = px.scatter(self.df, x=feature_name_x, y=feature_name_y, color=self.target_column, size=feature_name_x, hover_data=[self.target_column])
        fig.update_layout(margin=dict(l=20, r=50, t=50, b=50),
                         title=feature_name_x.capitalize() + " vs " + feature_name_y.capitalize() ,
                         xaxis_title=feature_name_x.capitalize(),
                         yaxis_title=feature_name_y.capitalize(),
                         legend_title=self.convert[self.target_column])
        
        # save the figure
        fig.write_image(f'charts/eda/correlation_distribution_scatter/{feature_name_x} vs {feature_name_y}.jpeg')
        

    
    def feature_correlation_catplot(self, feature_name_categorical, feature_name_numerical, kind="bar", height=7, aspect=2):
        """
        To plot correlation between a categorical and a numberical feature
        """
        
        sns.set(font_scale=1.2)

        # Visualize the correlation between a categorical and a numerical feature
        sns.catplot(data=self.df, x=feature_name_categorical, y=feature_name_numerical,
                    hue=self.target_column, kind=kind, height=height, aspect=2)
        plt.xticks(rotation=60)
        plt.xlabel(feature_name_categorical.capitalize())
        plt.ylabel(feature_name_numerical.capitalize())
        
        # save the figure
        plt.savefig(f'charts/eda/correlation_distribution_categorical/{feature_name_categorical} vs {feature_name_numerical}_{kind}.jpeg', bbox_inches='tight')

    
    
def display(message="Display Message:", df=None):
    """
    Display a message on a console
    """

    print(message)
    print(df)
    print("-------------------------------------------------------------------")