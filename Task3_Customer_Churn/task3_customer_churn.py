# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# load dataset
df = pd.read_csv("churn_modelling.csv")


# show first rows
print("\nFirst 5 Rows:\n")
print(df.head())


# dataset info
print("\nDataset Info:\n")
print(df.info())


# remove unnecessary columns
df.drop(
    ["RowNumber", "CustomerId", "Surname"],
    axis=1,
    inplace=True
)


# encode categorical columns
encoder = LabelEncoder()

df["Gender"] = encoder.fit_transform(df["Gender"])
df["Geography"] = encoder.fit_transform(df["Geography"])


# check missing values
print("\nMissing Values:\n")
print(df.isnull().sum())


# churn distribution
plt.figure(figsize=(6, 5))

sns.countplot(x="Exited", data=df)

plt.title("Customer Churn Distribution")

plt.savefig("churn_distribution.png")

plt.show()


# age vs churn
plt.figure(figsize=(8, 6))

sns.boxplot(
    x="Exited",
    y="Age",
    data=df
)

plt.title("Age vs Churn")

plt.savefig("age_vs_churn.png")

plt.show()


# correlation heatmap
plt.figure(figsize=(12, 8))

sns.heatmap(df.corr(), annot=True)

plt.title("Feature Correlation Heatmap")

plt.savefig("correlation_heatmap.png")

plt.show()


# split features and target
X = df.drop("Exited", axis=1)
y = df["Exited"]


# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# train model
model = RandomForestClassifier(random_state=42)

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


# feature importance
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)


# feature importance plot
plt.figure(figsize=(10, 6))

sns.barplot(
    x="Importance",
    y="Feature",
    data=importance
)

plt.title("Feature Importance")

plt.savefig("feature_importance.png")

plt.show()


print("\nTask 3 completed successfully.")