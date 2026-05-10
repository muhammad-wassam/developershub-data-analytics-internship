# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


# load dataset
df = pd.read_csv("insurance.csv")


# show first rows
print("\nFirst 5 Rows:\n")
print(df.head())


# dataset info
print("\nDataset Info:\n")
print(df.info())


# check missing values
print("\nMissing Values:\n")
print(df.isnull().sum())


# encode categorical columns
encoder = LabelEncoder()

df["sex"] = encoder.fit_transform(df["sex"])
df["smoker"] = encoder.fit_transform(df["smoker"])
df["region"] = encoder.fit_transform(df["region"])


# age vs charges
plt.figure(figsize=(8, 6))

sns.scatterplot(
    x="age",
    y="charges",
    data=df
)

plt.title("Age vs Insurance Charges")

plt.savefig("age_vs_charges.png")

plt.show()


# bmi vs charges
plt.figure(figsize=(8, 6))

sns.scatterplot(
    x="bmi",
    y="charges",
    data=df
)

plt.title("BMI vs Insurance Charges")

plt.savefig("bmi_vs_charges.png")

plt.show()


# smoker vs charges
plt.figure(figsize=(8, 6))

sns.boxplot(
    x="smoker",
    y="charges",
    data=df
)

plt.title("Smoking Status vs Insurance Charges")

plt.savefig("smoker_vs_charges.png")

plt.show()


# correlation heatmap
plt.figure(figsize=(10, 8))

sns.heatmap(df.corr(), annot=True)

plt.title("Feature Correlation Heatmap")

plt.savefig("correlation_heatmap.png")

plt.show()


# split features and target
X = df.drop("charges", axis=1)
y = df["charges"]


# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# train model
model = LinearRegression()

model.fit(X_train, y_train)


# predictions
y_pred = model.predict(X_test)


# MAE
mae = mean_absolute_error(y_test, y_pred)

print("\nMean Absolute Error:\n")
print(mae)


# RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nRoot Mean Squared Error:\n")
print(rmse)


# actual vs predicted plot
plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")

plt.title("Actual vs Predicted Charges")

plt.savefig("actual_vs_predicted.png")

plt.show()


print("\nTask 4 completed successfully.")