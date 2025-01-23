import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('lead_data.csv')

# Data Cleaning and Manipulation
# Handling duplicate data
data.drop_duplicates(inplace=True)

# Checking and handling NA values
data.dropna(inplace=True)

# Dropping unnecessary columns (assuming 'lead_id' is not required)
data.drop(columns=['lead_id'], inplace=True)

# EDA - Univariate and Bivariate Analysis
# Univariate Analysis
print(data['lead_converted'].value_counts())
sns.countplot(x='lead_converted', data=data)
plt.show()

# Bivariate Analysis
sns.pairplot(data, hue='lead_converted')
plt.show()

# Feature Scaling & Dummy Variables
# Encoding categorical variables
data = pd.get_dummies(data, drop_first=True)

# Define features (X) and target (y)
X = data.drop('lead_converted', axis=1)
y = data['lead_converted']

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Classification technique: Logistic Regression
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")

# Print the classification report with zero_division set to 0 to handle ill-defined metrics
print(classification_report(y_test, predictions, zero_division=0))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, predictions)
sns.heatmap(conf_matrix, annot=True, fmt='d')
plt.show()

# Conclusion and Recommendations
# Based on the results, we can identify the most promising leads