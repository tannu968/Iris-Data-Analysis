import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load the dataset
iris = load_iris()

# Create a DataFrame
df = pd.DataFrame(data=iris['data'], columns=iris['feature_names'])

# Add the target variable
df['target'] = iris['target']

# Add target names (optional, for better readability)
df['species'] = df['target'].apply(lambda x: iris['target_names'][x])

# Show the first 5 rows of the dataset
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Basic statistical summary
print(df.describe())

# Plotting relationships between variables
plt.scatter(df['sepal length (cm)'], df['sepal width (cm)'], c=df['target'])
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Sepal Length vs Width')
plt.show()
