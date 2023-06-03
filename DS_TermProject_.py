import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


# Step 2. Dataset
print("===================================================================")
print("                         Step 2. Dataset")
print("===================================================================")

df=pd.read_csv("adult.data",header=None)

features = ['age','workclass','fnlwgt','education',
            'education num','marital-status','occupation','relationship',
            'race','sex','capital-gain','capital-loss','hours-per-week','native-country','outcome']

# make columns.
df.columns=features
'''
# replace dirty value to NaN
for col in features:
    if df[col].dtype == 'object':
      df[col] = df[col].str.strip()
df.replace({"?":np.nan},inplace=True)

print(df.isna().sum())
'''
selected_features = ['age', 'sex', 'workclass', 'education','education num',
                     'capital-gain','capital-loss','hours-per-week','outcome']
data = df[selected_features]


print("----- data.head(5) -----\n", data.head(5))
print("\n----- data.shape -----\n",data.shape)
print("\n----- data.index -----\n",data.index)
print("\n----- data.columns -----\n",data.columns)



# Step 3. Missing Values
print("===================================================================")
print("                    Step 3. Missing Values")
print("===================================================================")

# in original data
print("\n----- df.isna().sum() -> in original data -----\n")
for col in features:
    if df[col].dtype == 'object':
      df[col] = df[col].str.strip()
df.replace({"?":np.nan},inplace=True)

print(df.isna().sum())

print("\n----- before : data.isna().sum() -> data used in use -----\n")
# replace dirty value to NaN
for col in selected_features:
    if data[col].dtype == 'object':
      data[col] = data[col].astype(str).str.strip()
        # data[col] = data[col].str.strip()

data.replace({"?":np.nan},inplace=True)

# check for missing values
print(data.isnull().sum())

print("\n\tdata['workclass']'s data : ",data['workclass'][27])

# replace missing values with median
data['age'].fillna(data['age'].median(), inplace=True)
data['sex'].fillna(data['sex'].mode()[0], inplace=True)
data['workclass'].fillna(data['workclass'].mode()[0], inplace=True)
data['education'].fillna(data['education'].mode()[0], inplace=True)
data['education num'].fillna(data['education num'].median(), inplace=True)
data['capital-gain'].fillna(data['capital-gain'].median(), inplace=True)
data['capital-loss'].fillna(data['capital-loss'].median(), inplace=True)
data['hours-per-week'].fillna(data['hours-per-week'].median(), inplace=True)

print("\n----- after : data.isna().sum() -----\n")
# check for missing values again
print(data.isnull().sum())

print("\n\tdata['workclass']'s data : ",data['workclass'][27])

print()
print(data.shape)

print()

# Keep real data.
real_data=data.copy()

# Step 4. Encoding Categorical Data
print("===================================================================")
print("              Step 4. Encoding Categorical Data")
print("===================================================================")

# Encoding 'sex' column
data = pd.get_dummies(data, columns=['sex'])

# Encoding 'workclass' column
data = pd.get_dummies(data, columns=['workclass'])

# Encoding 'education' column
data = pd.get_dummies(data, columns=['education'])

# Encoding 'outcome' column
data = pd.get_dummies(data, columns=['outcome'])

#pd.set_option('display.max_columns', None)

print("\n----- data after encoding -----\n")
print(data.head())

data_encoded = data


# Step 5. Clustering
print("===================================================================")
print("                        Step 5. Clustering")
print("===================================================================")

from sklearn.cluster import KMeans

# Define the number of clusters
n_clusters = 2

# Initialize the KMeans model
kmeans = KMeans(n_clusters=n_clusters, random_state=0)

# Fit the model to the data
kmeans.fit(data_encoded)

# Get the cluster labels
labels = kmeans.labels_

# Assign the cluster labels as the predicted outcome
data_encoded['predicted_outcome'] = labels

# Print the first few rows of the data
print(data_encoded.head())


# Step 6. Logistic Regression
print("===================================================================")
print("                 Step 6. Logistic Regression")
print("===================================================================")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Separate features and labels (assuming 'outcome' and 'working_hours' are the labels and the rest are features)
features = data_encoded.drop(['outcome_<=50K', 'outcome_>50K', 'hours-per-week'], axis=1)
labels_outcome = data_encoded['outcome_>50K']  # choose one of the outcome columns
labels_hours = data_encoded['hours-per-week']

# Split the data into training and test sets for outcome
features_train, features_test, labels_train, labels_test = train_test_split(features, labels_outcome, test_size=0.2, random_state=42)

# Initialize the Logistic Regression model for outcome
logreg_outcome = LogisticRegression()

# Fit the model to the training data
logreg_outcome.fit(features_train, labels_train)

# Predict the labels for the test set
predictions_outcome = logreg_outcome.predict(features_test)

# Calculate the accuracy of the model
accuracy_outcome = accuracy_score(labels_test, predictions_outcome)

print("\n----- Outcome Prediction Accuracy -----\n")
print(accuracy_outcome)


# Split the data into training and test sets for working hours
features_train, features_test, labels_train, labels_test = train_test_split(features, labels_hours, test_size=0.2, random_state=42)

# Initialize the Logistic Regression model for working hours
logreg_hours = LogisticRegression()

# Fit the model to the training data
logreg_hours.fit(features_train, labels_train)

# Predict the labels for the test set
predictions_hours = logreg_hours.predict(features_test)

# Calculate the accuracy of the model
accuracy_hours = accuracy_score(labels_test, predictions_hours)

print("\n----- Working Hours Prediction Accuracy -----\n")
print(accuracy_hours)



# Step 7. k-fold cross validation for testing classification
print("===================================================================")
print("                 Step 7. k-fold cross validation")
print("===================================================================")
from sklearn.model_selection import cross_val_score

# 1. Outcome

# Initialize the Logistic Regression model for outcome
logreg_outcome = LogisticRegression()

# Fit the model to the training data
logreg_outcome.fit(features_train, labels_train)

# Perform 10-fold cross-validation
scores_outcome = cross_val_score(logreg_outcome, features, labels_outcome, cv=10)

# Print the cross-validation scores
print("\n----- Outcome Prediction Cross-Validation Scores -----\n")
print(scores_outcome)

# Print the mean cross-validation score
print("\n----- Outcome Prediction Mean Cross-Validation Score -----\n")
print(np.mean(scores_outcome))

# 2. Working hours

# Initialize the Logistic Regression model for working hours
logreg_hours = LogisticRegression()

# Fit the model to the training data
logreg_hours.fit(features_train, labels_train)

# Perform 10-fold cross-validation
scores_hours = cross_val_score(logreg_hours, features, labels_hours, cv=10)

# Print the cross-validation scores
print("\n----- Working Hours Prediction Cross-Validation Scores -----\n")
print(scores_hours)

# Print the mean cross-validation score
print("\n----- Working Hours Prediction Mean Cross-Validation Score -----\n")
print(np.mean(scores_hours))



