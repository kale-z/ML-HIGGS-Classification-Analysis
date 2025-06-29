# Machine Learning Pipeline: <br> Feature Selection and Hyperparameter Optimization
In this project, I worked on the HIGGS Dataset, a large and high-dimensional dataset, to apply two critical components of the machine learning workflow: feature selection and hyperparameter tuning.

This project focuses on leveraging the **HIGGS Dataset**, a large and high-dimensional dataset, to demonstrate two critical components of the machine learning workflow: **feature selection** for dimensionality reduction and **hyperparameter tuning** for optimizing model performance in a classification task.


<br>


## Getting Started
This project implements a complete machine learning pipeline, with a core focus on robust feature selection and hyperparameter optimization. To run the project effectively, you will need to install the following essential Python packages:

### Dependencies
- [NumPy](https://github.com/numpy/numpy)
- [pandas](https://github.com/pandas-dev/pandas)
- [Scikit-Learn](https://github.com/scikit-learn/scikit-learn)
- [XGBoost](https://github.com/dmlc/xgboost)
- [Matplotlib](https://github.com/matplotlib/matplotlib)

<br>

### Dataset
This project utilizes the [**HIGGS Dataset**](https://archive.ics.uci.edu/dataset/280/higgs), which its primary purpose is a **classification problem**. The dataset file is `HIGGS.csv.gz`, with a size of 2.6 GB.

<br>

#### **Key Characteristics:**
* **Instances:** 11,000,000
* **Features:** 28
* **Feature Type:** Real
* **Associated Tasks:** Classification
* **Missing Values:** None

<br>

#### Feature Details:
* The first 21 features (columns 2-22) are **kinematic properties**.
* The last 7 features are **high-level features**, derived from the first 21 to enhance 
* The first column of the dataset is the **class label**: `1` for a signal event and `0` for a background event.

<br>

#### Data Overview
A snapshot of the dataset's structure, including column names, non-null counts, and data types, is provided below (representing 100,000 sampled entries):

| #   | Column                   | Non-Null Count | Dtype   |
| --- | ------------------------ | -------------- | ------- |
| 0   | class_label              | 100000         | float64 |
| 1   | lepton_pT                | 100000         | float64 |
| 2   | lepton_eta               | 100000         | float64 |
| 3   | lepton_phi               | 100000         | float64 |
| 4   | missing_energy_magnitude | 100000         | float64 |
| 5   | missing_energy_phi       | 100000         | float64 |
| 6   | jet_1_pt                 | 100000         | float64 |
| 7   | jet_1_eta                | 100000         | float64 |
| 8   | jet_1_phi                | 100000         | float64 |
| 9   | jet_1_b-tag              | 100000         | float64 |
| 10  | jet_2_pt                 | 100000         | float64 |
| 11  | jet_2_eta                | 100000         | float64 |
| 12  | jet_2_phi                | 100000         | float64 |
| 13  | jet_2_b-tag              | 100000         | float64 |
| 14  | jet_3_pt                 | 100000         | float64 |
| 15  | jet_3_eta                | 100000         | float64 |
| 16  | jet_3_phi                | 100000         | float64 |
| 17  | jet_3_b-tag              | 100000         | float64 |
| 18  | jet_4_pt                 | 100000         | float64 |
| 19  | jet_4_eta                | 100000         | float64 |
| 20  | jet_4_phi                | 100000         | float64 |
| 21  | jet_4_b-tag.             | 100000         | float64 |
| 22  | m_jj                     | 100000         | float64 |
| 23  | m_jjj                    | 100000         | float64 |
| 24  | m_lv                     | 100000         | float64 |
| 25  | m_jlv                    | 100000         | float64 |
| 26  | m_bb                     | 100000         | float64 |
| 27  | m_wbb                    | 100000         | float64 |
| 28  | m_wwbb                   | 100000         | float64 |


<br>


## Methods
### 1. Data Preprocessing
This section outlines the two primary preprocessing stages implemented: Outlier Analysis and Feature Scaling.

<br>

#### Outlier Analysis
Outliers in the dataset will be identified using the **Interquartile Range (IQR) method**. If detected, outliers will either be removed from the dataset or replaced with appropriate threshold values to mitigate their potential impact on model performance.

<br>

#### Feature Scaling
By applying **feature scaling**, all numerical variables in the dataset will be scaled to a `[0, 1] range` using the **MinMaxScaler**. This normalization process is critical for distance-based algorithms like KNN and for neural networks (MLP).

<br>

### 2. Feature Selection
To handle the high dimensionality of the HIGGS Dataset and enhance model performance, **filter-based feature selection** was applied. Using **Mutual Information (MI)**, which measures the dependency between features and the target variable, the **top 15 features** were selected from the original 28, to be used for all subsequent modeling as shown below:


| # | lepton_pT | lepton_eta | missing_energy_magnitude | jet_1_pt | jet_1_b-tag | jet_2_b-tag | jet_3_pt | jet_3_b-tag | jet_4_b-tag | m_jj | m_jjj | m_jlv | m_bb | m_wbb | m_wwbb | class_label |
| :---- | :-------- | :--------- | :----------------------- | :------- | :---------- | :---------- | :------- | :---------- | :---------- | :--- | :---- | :---- | :--- | :---- | :----- | :---------- |
| 3967303 | 0.443933  | 0.350940   | 0.084713                 | 0.816777 | 0.5         | 1.0         | 0.519267 | 0.0         | 0.0         | 1.000000 | 1.000000 | 0.294504 | 1.000000 | 0.929196 | 0.760714 | 0.0         |
| 5946179 | 0.066764  | 0.708684   | 0.479013                 | 1.000000 | 0.0         | 1.0         | 1.000000 | 0.0         | 0.0         | 0.676817 | 0.169489 | 1.000000 | 0.260321 | 0.701967 | 0.972739 | 1.0         |
| 6910558 | 0.443368  | 0.567027   | 0.338465                 | 0.154629 | 0.0         | 0.0         | 0.268240 | 1.0         | 1.0         | 0.403819 | 0.466301 | 0.374566 | 0.200412 | 0.301644 | 0.274806 | 0.0         |
| 3414332 | 0.568151  | 0.377951   | 0.195211                 | 0.143929 | 1.0         | 1.0         | 0.348305 | 0.0         | 0.0         | 0.134155 | 0.386085 | 0.390889 | 0.254794 | 0.261653 | 0.452096 | 0.0         |
| 5840458 | 0.353472  | 0.744898   | 0.650431                 | 0.675860 | 0.5         | 0.0         | 0.340376 | 0.0         | 1.0         | 1.000000 | 0.688773 | 0.612450 | 0.487501 | 0.582597 | 0.546620 | 1.0         |

<br>

### 3. Modeling and Evaluation
This section outlines the methodology for training machine learning models, optimizing their hyperparameters, and rigorously evaluating their performance using a robust nested cross-validation strategy.

<br>

#### Nested Cross-Validation
To ensure unbiased performance estimation and robust hyperparameter optimization tuning, a **nested cross-validation** approach was implemented:
* **Outer Loop:** 5-fold cross-validation.
* **Inner Loop:** 3-fold cross-validation.

<br>

#### Models Used and Hyperparameter Optimization
Hyperparameters for each model were systematically optimized within the inner cross-validation loop using `GridSearchCV`. Fixed model parameters were also set for reproducibility and specific algorithm requirements (e.g., `random_state=42`). The search spaces defined for optimization were:

* **KNN:** 
    * `n_neighbors=[3, 4, 5, 6, 7, 8, 9, 10, 11]`
* **SVM:** 
    * `C=[0.1, 1, 10]`
    * `kernel=['linear', 'rbf']`
    * `probability=True` (for ROC AUC calculation.)
* **MLP:** 
    * `hidden_layer_sizes=[(50,), (100,)]`
    * `activation=['relu', 'tanh']`
    * `max_iter=1000`
* **XGBoost:**
    * `n_estimators=[100, 200, 300]`
    * `learning_rate=[0.01, 0.05, 0.1]`
    * `max_depth=[3, 5, 7]`
    * `subsample=[0.7, 0.8, 0.9, 1.0]`
    * `colsample_bytree=[0.7, 0.8, 0.9, 1.0]`

<br>

#### Performance Metrics
* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC (Receiver Operating Characteristic - Area Under the Curve)

<br>

#### ROC Curves
Receiver Operating Characteristic (ROC) curves will be plotted for each model to visualize their trade-off between true positive rate and false positive rate across various threshold settings. The Area Under the Curve (AUC) scores will be calculated and interpreted to quantify the overall discriminative power of each model.


<br>


## Results
### Average Performance Metrics

The table below shows the average performance metrics for each model across all five outer folds, providing an unbiased estimate of their generalization ability:

| Model       | Accuracy | Precision | Recall | F1 Score | ROC AUC |
| :---------- | :------- | :-------- | :------ | :-------- | :-------- |
| **KNN** | 0.6670   | 0.6670    | 0.7381  | 0.7007    | 0.7275    |
| **SVM** | 0.7133   | 0.7103    | 0.7720  | 0.7399    | 0.7822    |
| **MLP** | 0.7207   | 0.7362    | 0.7377  | 0.7357    | 0.8004    |
| **XGBoost** | 0.7301   | 0.7416    | 0.7504  | 0.7460    | 0.8096    |

XGBoost achieved the highest overall performance with an average ROC AUC of 0.8096.

<br>

### Best Hyperparameters
The "best" hyperparameters for each model were determined during the inner cross-validation loop of each outer fold. For KNN, SVM, and MLP, the same optimal set of hyperparameters was consistently chosen across all folds. For XGBoost, while there was some variation across folds, the specific set of parameters that resulted in its highest single-fold ROC AUC is highlighted below.

* **K-Nearest Neighbors (KNN)**
    * `n_neighbors=11`

* **Support Vector Machine (SVM)**
    * `C=10`
    * `kernel='rbf'`

* **Multi-Layer Perceptron (MLP)**
    * `activation='relu'`
    * `hidden_layer_sizes=(100,)`

* **XGBoost**
    * `n_estimators`: `300`
    * `learning_rate`: `0.05`
    * `max_depth`: `7`
    * `subsample`: `0.9`
    * `colsample_bytree`: `0.7`
    

<br>

<p>
   <em>ROC Curves and AUC Scores: Comparison of KNN, SVM, MLP, and XGBoost Models</em>
   <br><br>
   <img style="max-width: 100%;height: 450px;" src="/plot.png" alt>
</p>

<br>

Based on the average performance metrics, **XGBoost** emerged as the best-performing model, outperforming others with an average ROC AUC of `0.8096` across the nested cross-validation process.

