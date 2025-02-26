import numpy as np
import matplotlib.pyplot as plt


from xgboost import XGBClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectFromModel

from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel

THREADS = 64

def xgboost_train_accur(X_train, y_train, X_test, y_test, device):

    pipe = make_pipeline(XGBClassifier(n_jobs=THREADS if device == "cpu" else None, tree_method="gpu_hist" if device == "cpu" else "hist"))
    
    cv_scores = cross_val_score(pipe, X_train.cpu(), y_train.cpu(), cv=5, scoring='accuracy')
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    pipe.fit(X_train.cpu(), y_train.cpu())
    train_accuracy = accuracy_score(y_train.cpu(), pipe.predict(X_train.cpu()))
    test_accuracy = accuracy_score(y_test.cpu(), pipe.predict(X_test.cpu()))
    return test_accuracy, cv_mean

def xgboost_accur_select_features(X_train, X_test, y_train, y_test, sorted_indices, baseline_accuracy, feat_step, device, feat_removal = False):
    cv_accur_arr = []
    test_accur_arr = []
 
    num_feat = range(1,len(sorted_indices),feat_step)
    num_feat_plot = []
    for N in num_feat:
        if feat_removal == False:
            select_feat = list(sorted_indices[:N])
        else:
            select_feat = list(sorted_indices[N:])
        num_feat_plot.append(N)        
       
       
        X_train_select_feat = X_train[:, select_feat] 
        X_test_select_feat = X_test[:, select_feat]
        test_accuracy, cv_accur_mean = xgboost_train_accur(X_train_select_feat, y_train, X_test_select_feat, y_test, device)
        cv_accur_arr.append(cv_accur_mean)
        test_accur_arr.append(test_accuracy)

    return cv_accur_arr,  test_accur_arr, num_feat_plot 


def mutual_info_features(X_train, y_train, X_train_column_names, contin_flag = False):
    if contin_flag == False:
        mutual_info = mutual_info_classif(X_train, y_train)
    else:
        mutual_info = mutual_info_regression(X_train, y_train)
    
    sorted_indices = np.argsort(mutual_info)[::-1] 
    sorted_mi = [mutual_info[i] for i in sorted_indices]
    sorted_names = [X_train_column_names[i] for i in sorted_indices]

    return sorted_indices, sorted_mi, sorted_names

def random_forest_features(X_train, y_train, X_train_column_names, contin_flag = False):

    if contin_flag == False:
        # Train a Random Forest model
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
    else:    
        rf = RandomForestRegressor(n_estimators=100, random_state=42)

    rf.fit(X_train, y_train)

    # Get feature importances
    importances = rf.feature_importances_
    print("Feature importances:", importances)
    print(len(importances))

    # Select features based on importance threshold
    selector = SelectFromModel(rf, threshold='mean', prefit=True)
    X_selected = selector.transform(X_train)

    print(f"Original feature count: {X_train.shape[1]}, Selected feature count: {X_selected.shape[1]}")
    

    plt.figure(figsize=(10, 2))
    plt.bar(range(X_train.shape[1]), importances)
    plt.xlabel("Feature Index")
    plt.ylabel("Importance")
    plt.ylim([0, max(importances)])
    plt.show()

    sorted_indices = np.argsort(importances)[::-1]  # Reverse the order to get descending sort

    # Step 2: Use the sorted indices to get the sorted importances and corresponding names
    sorted_importances = [importances[i] for i in sorted_indices]
    sorted_names = [X_train_column_names[i] for i in sorted_indices]

    return sorted_indices, sorted_importances, sorted_names

def svc_features(X_train, y_train, X_train_column_names):
    scaler = MaxAbsScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    #X_test_scaled = scaler.transform(X_test)

    # Step 3: Fit the LinearSVC model with L1 penalty (for feature selection)
    svm = LinearSVC(C=0.01, penalty='l1', dual=False, max_iter=5000)
    svm.fit(X_train, y_train)

    # Step 4: Extract the absolute feature importance (model coefficients)
    feature_importance = np.abs(svm.coef_.ravel())

    # Step 5: Sort the indices by feature importance (from highest to lowest)
    sorted_indices = np.argsort(feature_importance)[::-1]  # Reverse order for descending

    sorted_importances = [feature_importance[i] for i in sorted_indices]
    sorted_names = [X_train_column_names[i] for i in sorted_indices]

    return sorted_indices, sorted_importances, sorted_names
