import numpy as np
import matplotlib.pyplot as plt

import shap
from xgboost import XGBClassifier, XGBRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectFromModel

from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel

THREADS = 64

def xgboost_train_accur(X_train, y_train, X_test, y_test, device):
    """
    Trains XGBoost for the specified X/y train and test data.
    Returns dictionaries with training accuracy measures calculated for cross-validation and test.
    """
    # Initialize training pipelina
    pipe = make_pipeline(XGBClassifier(n_jobs=THREADS if device == "cpu" else None, tree_method="gpu_hist" if device == "cpu" else "hist"))

    # Run cross-validation and collect the accuracy metrics
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }   
    cv_results = cross_validate(pipe, X_train.cpu(), y_train.cpu(), cv=5, scoring=scoring, return_train_score=False)
    cv_accuracy_scores = {
        'accuracy': np.mean(cv_results['test_accuracy']),
        'precision': np.mean(cv_results['test_precision']),
        'recall': np.mean(cv_results['test_recall']),
        'f1': np.mean(cv_results['test_f1']),
        'roc_auc': np.mean(cv_results['test_roc_auc']),
    }

    # Fit on full training set
    pipe.fit(X_train.cpu(), y_train.cpu())

    # Test set predictions
    y_pred = pipe.predict(X_test.cpu())
    y_prob = pipe.predict_proba(X_test.cpu())[:, 1] if len(np.unique(y_train.cpu())) == 2 else None  # binary case

    # Collect final metrics on test set
    test_accuracy_scores = {
        'accuracy': accuracy_score(y_test.cpu(), y_pred),
        'precision': precision_score(y_test.cpu(), y_pred, zero_division=0),
        'recall': recall_score(y_test.cpu(), y_pred, zero_division=0),
        'f1': f1_score(y_test.cpu(), y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test.cpu(), y_prob)
    }
    return cv_accuracy_scores, test_accuracy_scores


def xgboost_accur_select_features(X_train, X_test, y_train, y_test, sorted_indices, feat_step, device, feat_removal = False):
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
        cv_accuracy_scores, test_accuracy_scores = xgboost_train_accur(X_train_select_feat, y_train, X_test_select_feat, y_test, device)
        cv_accur_arr.append(cv_accuracy_scores)
        test_accur_arr.append(test_accuracy_scores)

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
    # print("Feature importances:", importances)
    # print(len(importances))

    # Select features based on importance threshold
    selector = SelectFromModel(rf, threshold='mean', prefit=True)
    X_selected = selector.transform(X_train)

    print(f"Original feature count: {X_train.shape[1]}, Selected feature count: {X_selected.shape[1]}")
    

    # plt.figure(figsize=(10, 2))
    # plt.bar(range(X_train.shape[1]), importances)
    # plt.xlabel("Feature Index")
    # plt.ylabel("Importance")
    # plt.ylim([0, max(importances)])
    # plt.show()

    sorted_indices = np.argsort(importances)[::-1]  # Reverse the order to get descending sort

    # Step 2: Use the sorted indices to get the sorted importances and corresponding names
    sorted_importances = [importances[i] for i in sorted_indices]
    sorted_names = [X_train_column_names[i] for i in sorted_indices]

    return sorted_indices, sorted_importances, sorted_names

def shap_features(X_train, y_train, X_column_names, device, contin_flag = False):
    if contin_flag == False:
        pipe = make_pipeline(
            XGBClassifier(
                n_jobs=THREADS if device == "cpu" else None,
                tree_method="gpu_hist" if device != "cpu" else "hist"
            )
        )
        pipe.fit(X_train.cpu(), y_train.cpu())
        model = pipe.named_steps['xgbclassifier']
    else:    
        model = XGBRegressor(
        n_jobs=-1,                # Use all CPU cores
        tree_method="hist",   # Use "hist" for CPU, "gpu_hist" for GPU
        objective="reg:squarederror",  # Default loss function for regression
        )
        model.fit(X_train.cpu(), y_train.cpu())

    # 1. Train the pipeline

  #  pipe.fit(X_train.cpu(), y_train.cpu())

    # 2. Extract trained model
   # model = pipe.named_steps['xgbclassifier']

    # 3. Convert X to numpy
    X_np = X_train.cpu().numpy()

    # 4. SHAP explainer
    explainer = shap.Explainer(model, X_np)  # can also use shap.TreeExplainer(model)
    shap_values = explainer(X_np)

    # 5. Extract SHAP values and compute mean absolute SHAP value per feature
    shap_vals = shap_values.values  # shape: [n_samples, n_features]
    abs_shap_vals = np.abs(shap_vals)
    mean_abs_shap_vals = np.mean(abs_shap_vals, axis=0)

    # 6. Sort features by importance
    sorted_indices = np.argsort(mean_abs_shap_vals)[::-1]
    sorted_importances = mean_abs_shap_vals[sorted_indices]
    sorted_names = [X_column_names[i] for i in sorted_indices]

    # 7. Summary plot
    plt.figure(figsize=(5, 5))
    shap.summary_plot(shap_vals, X_np, feature_names=X_column_names)
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


def plot_accuracy_metric(metric, test_accuracy_scores, cv_accuracy_scores, test_accur_arr, test_accur_arr_rem, cv_accur_arr, cv_accur_arr_rem, num_feat):
    plt.axhline(y=test_accuracy_scores[metric], color='darkred', linestyle='--', linewidth=1.5, label='baseline test')
    plt.axhline(y=cv_accuracy_scores[metric], color='darkblue', linestyle='--', linewidth=1.5, label='baseline CV')

    plt.plot(num_feat, [scores[metric] for scores in test_accur_arr], c = "tab:red", label = "test | add")
    plt.plot(num_feat, [scores[metric] for scores in cv_accur_arr], c = "tab:blue", label = "cv | add")

    plt.plot(num_feat,  [scores[metric] for scores in test_accur_arr_rem], c = "tab:red", label = "test | remove", alpha = 0.5)
    plt.plot(num_feat, [scores[metric] for scores in cv_accur_arr_rem], c = "tab:blue", label = "cv | remove", alpha = 0.5)

    plt.xlabel("number of features added/removed")
    plt.ylabel(metric)