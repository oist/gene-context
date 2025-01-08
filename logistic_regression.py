import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from utils.utils import read_xy_data

from sklearn.preprocessing import MaxAbsScaler

import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score

from set_transformer.main import read_ogt_data


phenotype = "ogt" # "aerob"

if phenotype == "aerob":
    data_filename_train = "data/all_gene_annotations.added_incompleteness_and_contamination.training.tsv"#'data/all_gene_annotations.added_incompleteness_and_contamination.training.tsv'
    data_filename_test =  "data/all_gene_annotations.added_incompleteness_and_contamination.testing.tsv"#"data/all_gene_annotations.added_incompleteness_and_contamination.testing.tsv"
    y_filename = "data/bacdive_scrape_20230315.json.parsed.anaerobe_vs_aerobe.with_cyanos.csv"
    d3_train, X_train, y_train = read_xy_data(data_filename_train, y_filename)

    if 'family_right' in X_train.columns:
        X_train = X_train.drop(columns=['family_right'])

    if 'phylum_right' in X_train.columns:
        X_train = X_train.drop(columns=['phylum_right'])

    if 'class_right' in X_train.columns:
        X_train = X_train.drop(columns=['class_right'])

    if 'order_right' in X_train.columns:
        X_train = X_train.drop(columns=['order_right'])        

    if 'genus_right' in X_train.columns:
        X_train = X_train.drop(columns=['genus_right'])  

    X_train = X_train.values
    y_train = y_train.values
    y_train = y_train.ravel()

    d3_test, X_test, y_test = read_xy_data(data_filename_test, y_filename)


    if 'family_right' in X_test.columns:
        X_test = X_test.drop(columns=['family_right'])

    if 'phylum_right' in X_test.columns:
        X_test = X_test.drop(columns=['phylum_right'])

    if 'class_right' in X_test.columns:
        X_test = X_test.drop(columns=['class_right'])

    if 'order_right' in X_test.columns:
        X_test = X_test.drop(columns=['order_right'])        

    if 'genus_right' in X_test.columns:
        X_test = X_test.drop(columns=['genus_right'])  

    X_test = X_test.values
    y_test = y_test.values
    y_test = y_test.ravel()

    scaler = MaxAbsScaler()

    # Fit and transform the data
    X_train = scaler.fit_transform(X_train)

elif phenotype == "ogt":
    X_train, y_train, X_test, y_test, num_classes = read_ogt_data()
    X_train = X_train.cpu().numpy()
    y_train = y_train.cpu().numpy()
    X_test = X_test.cpu().numpy() 
    y_test = y_test.cpu().numpy()


print(X_train)

print(f"X_train shape = {X_train.shape}")
print(f"y_train shape = {y_train.shape}")

print(f"X_test shape = {X_test.shape}")
print(f"y_test shape = {y_test.shape}")

# Create and train the logistic regression model
num_iter = 100
model = LogisticRegression(max_iter=num_iter)
model.fit(X_train, y_train)



scores = cross_val_score(model, X_train, y_train, cv=5)  # 5-fold cross-validation
print(f'Cross-validation scores: {scores}')


logits = model.decision_function(X_test)  # This gives the raw logits

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Create values for sigmoid curve
logit_range = np.linspace(logits.min(), logits.max(), 500)  # Range of logits for smooth curve
sigmoid_values = sigmoid(logit_range)  # Apply sigmoid

if phenotype == "aerob":
    plt.figure()
    counts, bins, patches = plt.hist(logits, bins=50, alpha=0.7, color='blue', label='Logit Distribution')

    # Normalize the heights to have the maximum value equal to 1
    counts_normalized = counts / counts.max()

    # Rescale the histogram
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    plt.cla()  # Clear the current axes
    plt.bar(bin_centers, counts_normalized, width=(bins[1] - bins[0]), alpha=0.7, color='blue', label='Logit Distribution (rescaled)')

    plt.plot(logit_range, sigmoid_values, color='red', linewidth=2, label='Sigmoid Function')
    plt.title(f"# iterations = {num_iter}")
    plt.legend()



# Get predicted probabilities
probabilities = model.predict_proba(X_test)

# Print probabilities
print("Predicted probabilities for each class:")
print(probabilities)


# Predict on the test set
y_pred = model.predict(X_test)

print(f"y_pred = {y_pred}")


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



# df = pd.DataFrame(X_train)
# correlation_matrix = df.corr()

# # Plot the correlation matrix
# plt.figure(figsize=(10, 8))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
# plt.title('Correlation Matrix')
plt.show()