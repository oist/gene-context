import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from utils import read_xy_data

# Generate a purely random dataset
# Generate 1000 samples with 20 random features
#X = np.random.randn(1000, 20)  # 1000 samples, 20 features from a normal distribution
#y = np.random.randint(0, 2, 1000)  # 1000 binary target values (0 or 1)

# Split the dataset into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# print(f"X_train shape = {X_train.shape}")
# print(f"y_train shape = {y_train.shape}")

data_filename_train = 'data/all_gene_annotations.added_incompleteness_and_contamination.training.tsv'
data_filename_test = "data/all_gene_annotations.added_incompleteness_and_contamination.testing.tsv"
y_filename = "data/bacdive_scrape_20230315.json.parsed.anaerobe_vs_aerobe.with_cyanos.csv"
d3_train, X_train, y_train = read_xy_data(data_filename_train, y_filename)

X_train = X_train.values
y_train = y_train.values
y_train = y_train.ravel()

d3_test, X_test, y_test = read_xy_data(data_filename_test, y_filename)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
#X_train = pca.fit_transform(X_train)
#X_train = X_train.T

print("Data after PCA reduction:")
print(X_train.shape)

X_test = X_test.values
y_test = y_test.values
y_test = y_test.ravel()

#X_test = pca.fit_transform(X_test)

print(f"X_train shape = {X_train.shape}")
print(f"y_train shape = {y_train.shape}")

# Create and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Display results
print(f"Accuracy: {accuracy:.4f}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)
