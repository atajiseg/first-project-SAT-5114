#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Load the dataset
file_path = "oasis_cross-sectional.csv"
data = pd.read_csv(file_path)

# Remove rows where the target variable 'CDR' is missing
data_clean = data.dropna(subset=['CDR'])

# Assuming you want to keep specific columns for X. Adjust the list based on your dataset's relevant features.
# This is an example based on common features you might be interested in. Adjust as necessary.
feature_columns = ['nWBV', 'ASF'] # Add other feature columns as per your analysis needs

# Define the features matrix X and target variable y
X = data_clean[feature_columns]
y = data_clean['CDR']

# Check if X is empty
if X.empty:
    raise ValueError("The feature matrix X is empty after preprocessing.")

# Impute missing values in the features
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Define the grid of hyperparameters to search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt'],
    'bootstrap': [True, False]
}

# Initialize the classifier
clf = RandomForestClassifier()

# Initialize the GridSearchCV object
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Print the best parameters found by the grid search
print("Best parameters:", grid_search.best_params_)


# In[ ]:





# In[ ]:




