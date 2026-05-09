# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# load iris dataset
iris = sns.load_dataset("iris")


# show first rows
print("\nFirst 5 Rows:\n")
print(iris.head())


# dataset shape
print("\nDataset Shape:\n")
print(iris.shape)


# column names
print("\nColumns:\n")
print(iris.columns)


# dataset information
print("\nDataset Info:\n")
print(iris.info())


# statistical summary
print("\nStatistical Summary:\n")
print(iris.describe())


# check missing values
print("\nMissing Values:\n")
print(iris.isnull().sum())


# scatter plot
plt.figure(figsize=(8, 6))

sns.scatterplot(
    x="sepal_length",
    y="petal_length",
    hue="species",
    data=iris
)

plt.title("Sepal Length vs Petal Length")

plt.show()


# histogram
plt.figure(figsize=(8, 6))

sns.histplot(
    iris["sepal_length"],
    bins=20,
    kde=True
)

plt.title("Distribution of Sepal Length")
plt.xlabel("Sepal Length")

plt.show()


# box plot
plt.figure(figsize=(8, 6))

sns.boxplot(
    x="species",
    y="sepal_length",
    data=iris
)

plt.title("Box Plot of Sepal Length")

plt.show()


# pair plot
sns.pairplot(
    iris,
    hue="species"
)

plt.show()


print("\nTask 1 completed successfully.")