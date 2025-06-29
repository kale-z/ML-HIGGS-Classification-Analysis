import os
import time
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from datetime import datetime
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve


### start time
start_time = time.time()
start_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(f"Start Time: {start_time_str}\n")

### Suppress infos and warnings logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  



########################################################################################
# Load the dataset and specify column names 
########################################################################################



# Define the path to your HIGGS.csv file
file_path = 'HIGGS.csv'

# Define the column names as per the dataset's description 
# The first column is the class label (target), followed by 28 features.
column_names = [
    'class_label',
    'lepton_pT', 'lepton_eta', 'lepton_phi',
    'missing_energy_magnitude', 'missing_energy_phi',
    'jet_1_pt', 'jet_1_eta', 'jet_1_phi', 'jet_1_b-tag',
    'jet_2_pt', 'jet_2_eta', 'jet_2_phi', 'jet_2_b-tag',
    'jet_3_pt', 'jet_3_eta', 'jet_3_phi', 'jet_3_b-tag',
    'jet_4_pt', 'jet_4_eta', 'jet_4_phi', 'jet_4_b-tag',
    'm_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb'
]

print(f"\nLoading the dataset from with specified column names...")

# Load the CSV file, explicitly stating there is no header and providing names
higgs_df = pd.read_csv(file_path, header=None, names=column_names)



########################################################################################
# Take 100.000 random samples
########################################################################################



sample_size = 100000
print("########################################################################################")
print(f"Taking a random sample of {sample_size} rows from the dataset...")

# Use .sample() to get a random subset of the DataFrame
higgs_sampled_df = higgs_df.sample(n=sample_size, random_state=42)

print("Sampled dataset loaded successfully with correct column names!")
print(f"Shape of the dataset: {higgs_sampled_df.shape}")

print("\nFirst 5 rows of the dataset with correct headers:")
print(higgs_sampled_df.head())

print("\n--- DataFrame Info (with correct columns) ---")
higgs_sampled_df.info()

print("\n--- Descriptive Statistics (with correct columns) ---")
print(higgs_sampled_df.describe())



########################################################################################
# Perform outlier analysis using the IQR method
########################################################################################



print("########################################################################################")
print("\nPerforming outlier analysis using the IQR method...")

# Identify feature columns (all columns except 'class_label')
feature_columns = [col for col in higgs_sampled_df.columns if col != 'class_label']

# Dictionary to store outlier summary for each feature (optional, but useful for later review)
outlier_summary = {}

for col in feature_columns:
    Q1 = higgs_sampled_df[col].quantile(0.25)
    Q3 = higgs_sampled_df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Count outliers
    # Outliers are values that fall below the lower bound or above the upper bound
    num_outliers_lower = higgs_sampled_df[higgs_sampled_df[col] < lower_bound].shape[0]
    num_outliers_upper = higgs_sampled_df[higgs_sampled_df[col] > upper_bound].shape[0]
    num_outliers = num_outliers_lower + num_outliers_upper

    percentage_outliers = (num_outliers / len(higgs_sampled_df)) * 100

    outlier_summary[col] = {
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR,
        'Lower Bound': lower_bound,
        'Upper Bound': upper_bound,
        'Number of Outliers': num_outliers,
        'Percentage of Outliers': percentage_outliers
    }

    print(f"\n--- Feature: {col} ---")
    print(f"  Q1: {Q1:.4f}")
    print(f"  Q3: {Q3:.4f}")
    print(f"  IQR: {IQR:.4f}")
    print(f"  Lower Bound: {lower_bound:.4f}")
    print(f"  Upper Bound: {upper_bound:.4f}")
    print(f"  Number of Outliers: {num_outliers} ({percentage_outliers:.2f}%)")

print("\nOutlier analysis complete for all features.")

# The 'outlier_summary' dictionary now contains the details for each feature.
# You can inspect it further if needed, e.g., print(outlier_summary['lepton_pT'])



########################################################################################
# Handling outliers by replacing them with IQR-based threshold values
########################################################################################



print("########################################################################################")
print("\nHandling outliers by replacing them with IQR-based threshold values...")

# Identify feature columns (all columns except 'class_label')
feature_columns = [col for col in higgs_sampled_df.columns if col != 'class_label']

for col in feature_columns:
    Q1 = higgs_sampled_df[col].quantile(0.25)
    Q3 = higgs_sampled_df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Replace values below the lower bound with the lower bound
    higgs_sampled_df[col] = np.where(higgs_sampled_df[col] < lower_bound, lower_bound, higgs_sampled_df[col])
    
    # Replace values above the upper bound with the upper bound
    higgs_sampled_df[col] = np.where(higgs_sampled_df[col] > upper_bound, upper_bound, higgs_sampled_df[col])

print("Outlier handling complete for all features in higgs_sampled_df.")


'''
You can optionally re-run the outlier analysis code after this step to verify by uncommenting the below lines
that the number of outliers for each feature is now 0.
'''

# print("Checking if outliers have been handled...")
# for col in feature_columns:
#     Q1 = higgs_sampled_df[col].quantile(0.25)
#     Q3 = higgs_sampled_df[col].quantile(0.75)
#     IQR = Q3 - Q1
#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR
#     num_outliers = len(higgs_sampled_df[(higgs_sampled_df[col] < lower_bound) | (higgs_sampled_df[col] > upper_bound)])
#     print(f"  Verified outliers for {col}: {num_outliers}")



########################################################################################
# Perform feature scaling using MinMaxScaler normalization 
########################################################################################


print("########################################################################################")
print("\nApplying MinMaxScaler to scale all numerical features to the [0, 1] range...")

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Apply MinMaxScaler to the feature columns
# We fit the scaler to the data and then transform it.
# This operation will overwrite the original feature columns in higgs_sampled_df
higgs_sampled_df[feature_columns] = scaler.fit_transform(higgs_sampled_df[feature_columns])

print("Feature scaling complete! Displaying the first 5 rows of the scaled features:")
# Display head of the DataFrame to show scaled values for features
print(higgs_sampled_df.head())

''' Optional: Verify the min and max of a scaled feature (should be close to 0 and 1) by uncommenting the below lines '''
# print("\nVerification: Min and Max values for 'lepton_pT' after scaling:")
# print(f"Min: {higgs_sampled_df['lepton_pT'].min():.4f}")
# print(f"Max: {higgs_sampled_df['lepton_pT'].max():.4f}")



########################################################################################
# Applying Mutual Information for feature selection to select the top 15 features
########################################################################################



print("########################################################################################")
print("\nApplying Mutual Information for feature selection to select the top 15 features...")

# Separate features (X) and target (y)
X = higgs_sampled_df[feature_columns]
y = higgs_sampled_df['class_label']

# Initialize SelectKBest with mutual_info_classif and specify k=15 for top 15 features
selector = SelectKBest(mutual_info_classif, k=15)

# Fit the selector to the data and transform X
X_selected = selector.fit_transform(X, y)

# Get the names of the selected features
selected_feature_indices = selector.get_support(indices=True)
selected_feature_names = [feature_columns[i] for i in selected_feature_indices]

print(f"Selected Top 15 Features based on Mutual Information:")
print(selected_feature_names)

# Update higgs_sampled_df to only include the selected features and the class_label
# Create a new DataFrame from the selected features and re-add the class_label
# Preserve the original index to correctly align with 'y'
higgs_sampled_df = pd.DataFrame(X_selected, columns=selected_feature_names, index=higgs_sampled_df.index)
higgs_sampled_df['class_label'] = y # Add the target column back

print("\nDataset updated with only the top 15 selected features.")
print(f"New shape of the dataset: {higgs_sampled_df.shape}")
print("\nFirst 5 rows of the dataset with selected features:")
print(higgs_sampled_df.head())



########################################################################################
# Performing Nested Cross-Validation for Modeling and Evaluation
########################################################################################



print("########################################################################################")
print("Starting Nested Cross-Validation for Modeling and Evaluation...")

# Separate features (X) and target (y)
X = higgs_sampled_df[selected_feature_names]
y = higgs_sampled_df['class_label']

# Convert DataFrame to NumPy arrays for scikit-learn compatibility
X_np = X.to_numpy()
y_np = y.to_numpy()

# Define models and their hyperparameter ranges for the inner loop
# Note: For SVM, 'probability=True' is needed to get ROC AUC scores
models_and_params = {
    'KNN': {
        'model': KNeighborsClassifier(),
        'params': {'n_neighbors': list(range(3, 12))} # n_neighbors between 3 and 11 
    },
    'SVM': {
        'model': SVC(probability=True, random_state=42), # probability=True for ROC AUC
        'params': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    },
    'MLP': {
        'model': MLPClassifier(random_state=42, max_iter=1000), # Increased max_iter for convergence
        'params': {'hidden_layer_sizes': [(50,), (100,)], 'activation': ['relu', 'tanh']}
    },
    'XGBoost': {
        'model': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42), # Suppress warning, set eval_metric
        'params': {
            'n_estimators': [100, 200, 300], 
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
        }
    }
}

# Store results from each outer fold
outer_fold_results = []
outer_roc_curves = {}

# Set up Outer Loop: 5-fold Stratified Cross-Validation
# StratifiedKFold ensures that the class distribution is maintained in each fold.
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print(f"Starting Outer Loop ({outer_cv.n_splits}-fold CV)...")
for outer_train_idx, outer_test_idx in outer_cv.split(X_np, y_np):
    X_outer_train, X_outer_test = X_np[outer_train_idx], X_np[outer_test_idx]
    y_outer_train, y_outer_test = y_np[outer_train_idx], y_np[outer_test_idx]

    print(f"\nOuter Fold: Training data shape {X_outer_train.shape}, Test data shape {X_outer_test.shape}")

    # Results for current outer fold across all models
    fold_models_results = {}

    # Iterate through each model
    for model_name, config in models_and_params.items():
        print(f"  Processing {model_name} in current Outer Fold...")

        classifier = config['model']
        param_grid = config['params']

        # Set up Inner Loop: 3-fold Stratified Cross-Validation for Hyperparameter Tuning
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        # GridSearchCV will perform the inner loop (3-fold CV)
        # It trains on inner_train and validates on inner_test
        grid_search = GridSearchCV(
            estimator=classifier,
            param_grid=param_grid,
            cv=inner_cv,
            scoring='roc_auc', # Use ROC AUC for hyperparameter selection
            n_jobs=-1, # Use all available CPU cores
            verbose=0 # Set to 1 or 2 for more detailed output
        )

        # Fit GridSearchCV on the outer training data (X_outer_train, y_outer_train)
        grid_search.fit(X_outer_train, y_outer_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_ # ROC AUC score from inner CV

        print(f"    {model_name}: Best Inner CV ROC AUC: {best_score:.4f} with params: {best_params}")

        # Evaluate the best model on the Outer Test set
        y_pred = best_model.predict(X_outer_test)
        y_prob = best_model.predict_proba(X_outer_test)[:, 1] # Probability of the positive class

        # Calculate performance metrics
        accuracy = accuracy_score(y_outer_test, y_pred)
        precision = precision_score(y_outer_test, y_pred)
        recall = recall_score(y_outer_test, y_pred)
        f1 = f1_score(y_outer_test, y_pred)
        roc_auc = roc_auc_score(y_outer_test, y_prob)

        print(f"    {model_name}: Outer Test Performance:")
        print(f"      Accuracy: {accuracy:.4f}")
        print(f"      Precision: {precision:.4f}")
        print(f"      Recall: {recall:.4f}")
        print(f"      F1 Score: {f1:.4f}")
        print(f"      ROC AUC: {roc_auc:.4f}")

        # Store results for this model in this outer fold
        fold_models_results[model_name] = {
            'best_params': best_params,
            'inner_cv_roc_auc': best_score,
            'outer_test_accuracy': accuracy,
            'outer_test_precision': precision,
            'outer_test_recall': recall,
            'outer_test_f1': f1,
            'outer_test_roc_auc': roc_auc,
            'y_true': y_outer_test,
            'y_prob': y_prob
        }

        # Store ROC curve data for later plotting (One-Vs-All is implicit for binary classification)
        fpr, tpr, thresholds = roc_curve(y_outer_test, y_prob)
        if model_name not in outer_roc_curves:
            outer_roc_curves[model_name] = []
        outer_roc_curves[model_name].append({'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc})


    outer_fold_results.append(fold_models_results)

print("\nNested Cross-Validation Completed.")


# Example of how to average ROC AUCs:
print("\nAverage ROC AUC across Outer Folds for each Model:")
for model_name in models_and_params.keys():
    avg_roc_auc = np.mean([f[model_name]['outer_test_roc_auc'] for f in outer_fold_results])
    print(f"- {model_name}: {avg_roc_auc:.4f}")

# The 'outer_roc_curves' dictionary contains lists of (fpr, tpr, roc_auc) for each outer fold.
# You would typically average these for a final plot or plot them individually.



########################################################################################
# Generating ROC Curves and Visualizing AUC Scores for Each Model
########################################################################################



print("########################################################################################")
print("\nGenerating ROC Curves and Visualizing AUC Scores...")


plt.figure(figsize=(10, 8))
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='grey', label='Random Classifier')

# Plot ROC curves for each model, showing individual curves from each outer fold
# and their average AUC score
for model_name, roc_data_list in outer_roc_curves.items():
    all_roc_aucs = []
    for i, fold_data in enumerate(roc_data_list):
        fpr = fold_data['fpr']
        tpr = fold_data['tpr']
        roc_auc = fold_data['roc_auc']
        all_roc_aucs.append(roc_auc)

        # Plot individual ROC curve for each fold
        plt.plot(fpr, tpr, lw=1.5, alpha=0.6,
                 label=f'{model_name} - Fold {i+1} (AUC = {roc_auc:.2f})')

    # Calculate and print the average AUC for the model
    avg_roc_auc = np.mean(all_roc_aucs)
    print(f"Average ROC AUC for {model_name}: {avg_roc_auc:.4f}")

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curves for Models')
plt.legend(loc='lower right', bbox_to_anchor=(1.0, 0.0))
plt.grid(True)
plt.tight_layout()
plt.show()
print("\nROC Curve plotting completed.")



# End time
end_time = time.time()
end_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(f"\nEnd Time: {end_time_str}")

# Elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed Time: {elapsed_time:.2f} seconds")
