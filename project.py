# Iris Dataset Analysis and Visualization Project

# Import libraries 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Set seaborn style for better visuals
sns.set(style="whitegrid")

def load_and_explore_data():
    """
    Load the Iris dataset, convert it to a DataFrame,
    and perform initial exploration.
    """
    try:
        # Load dataset
        iris = load_iris()
        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

        # Display first 5 rows
        print("First 5 rows of the dataset:")
        print(df.head(), "\n")

        # Data types and missing values
        print("Data types:")
        print(df.dtypes, "\n")

        print("Missing values per column:")
        print(df.isnull().sum(), "\n")

        return df

    except Exception as e:
        print(f"Error loading or exploring data: {e}")
        return None

def basic_data_analysis(df):
    """
    Perform basic statistics and grouping analysis.
    """
    try:
        print("Basic statistics of numerical columns:")
        print(df.describe(), "\n")

        print("Mean values grouped by species:")
        group_means = df.groupby('species').mean()
        print(group_means, "\n")

        print("Observations:")
        print("- Setosa species has the smallest petal length and width on average.")
        print("- Versicolor and Virginica have larger sepal and petal dimensions.\n")

        return group_means

    except Exception as e:
        print(f"Error in data analysis: {e}")
        return None

def create_visualizations(df, group_means):
    """
    Create four types of visualizations:
    1. Line chart
    2. Bar chart
    3. Histogram
    4. Scatter plot
    """
    try:
        # 1. Line Chart: Sepal length over sample index by species
        plt.figure(figsize=(10, 6))
        for species in df['species'].unique():
            subset = df[df['species'] == species]
            plt.plot(subset.index, subset['sepal length (cm)'], label=species)
        plt.title('Sepal Length Over Sample Index by Species')
        plt.xlabel('Sample Index')
        plt.ylabel('Sepal Length (cm)')
        plt.legend()
        plt.show()

        # 2. Bar Chart: Average petal length per species
        plt.figure(figsize=(8, 6))
        sns.barplot(x=group_means.index, y=group_means['petal length (cm)'])
        plt.title('Average Petal Length per Species')
        plt.xlabel('Species')
        plt.ylabel('Average Petal Length (cm)')
        plt.show()

        # 3. Histogram: Distribution of sepal width
        plt.figure(figsize=(8, 6))
        sns.histplot(df['sepal width (cm)'], bins=20, kde=True)
        plt.title('Distribution of Sepal Width')
        plt.xlabel('Sepal Width (cm)')
        plt.ylabel('Frequency')
        plt.show()

        # 4. Scatter Plot: Sepal length vs Petal length by species
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species')
        plt.title('Sepal Length vs Petal Length by Species')
        plt.xlabel('Sepal Length (cm)')
        plt.ylabel('Petal Length (cm)')
        plt.legend(title='Species')
        plt.show()

    except Exception as e:
        print(f"Error creating visualizations: {e}")

def main():
    df = load_and_explore_data()
    if df is not None:
        group_means = basic_data_analysis(df)
        if group_means is not None:
            create_visualizations(df, group_means)

if __name__ == "__main__":
    main()
