# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


# load dataset
df = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")


# show first rows
print("\nFirst 5 Rows:\n")
print(df.head())


# dataset info
print("\nDataset Info:\n")
print(df.info())


# missing values
print("\nMissing Values:\n")
print(df.isnull().sum())


# fill missing values
df["Gender"].fillna(df["Gender"].mode()[0], inplace=True)
df["Married"].fillna(df["Married"].mode()[0], inplace=True)
df["Dependents"].fillna(df["Dependents"].mode()[0], inplace=True)
df["Self_Employed"].fillna(df["Self_Employed"].mode()[0], inplace=True)

df["LoanAmount"].fillna(df["LoanAmount"].mean(), inplace=True)
df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mean(), inplace=True)
df["Credit_History"].fillna(df["Credit_History"].mean(), inplace=True)


# encode categorical columns
encoder = LabelEncoder()

categorical_columns = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
    "Loan_Status"
]

for column in categorical_columns:
    df[column] = encoder.fit_transform(df[column])


# drop loan id
df.drop("Loan_ID", axis=1, inplace=True)


# data visualization
plt.figure(figsize=(8, 6))

sns.histplot(df["ApplicantIncome"], bins=30)

plt.title("Applicant Income Distribution")

plt.savefig("income_distribution.png")

plt.show()


# correlation heatmap
plt.figure(figsize=(10, 8))

sns.heatmap(df.corr(), annot=True)

plt.title("Feature Correlation Heatmap")

plt.savefig("correlation_heatmap.png")

plt.show()


# split features and target
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]


# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# train model
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)


# predictions
y_pred = model.predict(X_test)


# accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:\n")
print(accuracy)


# confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:\n")
print(cm)


# confusion matrix plot
plt.figure(figsize=(6, 5))

sns.heatmap(cm, annot=True, fmt="d")

plt.title("Confusion Matrix")

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("confusion_matrix.png")

plt.show()


print("\nTask 2 completed successfully.")