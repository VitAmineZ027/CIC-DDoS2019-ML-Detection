# DDoS Anomaly Detection Using CIC-DDoS2019 Dataset
This project implements a DDoS anomaly detection pipeline using the CIC-DDoS2019 dataset. It leverages extensive Exploratory Data Analysis (EDA), robust data preprocessing, feature engineering, machine learning models, and a deep learning model to classify network traffic anomalies.

## Dataset

---

The dataset used for this project is the **CIC-DDoS2019** dataset, specifically the `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` file. This dataset contains labeled network traffic data, including both benign and DDoS attack instances, enabling supervised learning.

- **Kaggle Source**: The dataset was obtained from [Kaggle](https://www.kaggle.com/datasets/aymenabb/ddos-evaluation-dataset-cic-ddos2019/data), where it is pre-prepared and easy to download.
- **Original Source**: The dataset originates from the [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ddos-2019.html).

## Requirements

---

```bash
$ pip install -r requirements.txt
```
### Key Libraries:
- **Python** 3.8+
- **pandas**
- **numpy**
- **matplotlib**
- **seaborn**
- **scikit-learn**
- **xgboost**
- **tensorflow/keras**

## Exploratory Data Analysis (EDA)
---
Comprehensive EDA was performed to understand the dataset:

### Dataset Overview:
- Displayed dataset shape, column names, and data types.
- Identified and handled missing values.
- Replaced invalid numeric values (e.g., infinity) with appropriate alternatives.

### Visualizations:
- **Missing Values**: Bar plots for columns with missing values.
- **Numeric Feature Distribution**: Histograms of numeric features to analyze value distributions.
- **Boxplots**: Detected and visualized outliers in numeric features.
- **Target Variable Distribution**: Pie chart showing the class distribution of labels.

## Data Preprocessing
---
### Target Encoding:
- Label-encoded the target column (`Label`) to convert class names into numeric values.

### Categorical Encoding:
- Label-encoded all categorical features.

### Irrelevant Feature Removal:
- Dropped unnecessary columns like `Timestamp`.

### Handling Missing and Invalid Values:
- Imputed missing values with column means.
- Replaced infinite values with the maximum value within the feature.

### Data Normalization:
- Standardized features using `StandardScaler` for better model performance.

### Data Splitting:
- Split the dataset into training and test sets with an 80:20 ratio using `train_test_split`.


## Feature Engineering
---
### Feature Importance:
- Used `SelectKBest` with ANOVA F-value (`f_classif`) to identify the top 10 features.
- Determined feature importances using a Random Forest Classifier, visualized as a bar plot.

### Outlier Detection:
- Identified outliers via boxplots, providing insights into data anomalies.


## Models and Results
----
### Machine Learning Models
Three machine learning models were trained with hyperparameter optimization using grid search:

#### Support Vector Machine (SVM):
- **Hyperparameters Tuned**: 
  - `C` (regularization).
  - Kernel type (`linear`, `rbf`).
- **Performance**:
  - Achieved strong precision, recall, and F1 scores.
  - Competitive accuracy on DDoS detection.

#### Random Forest Classifier:
- **Hyperparameters Tuned**:
  - `n_estimators` (number of trees).
  - `max_depth` (tree depth).
- **Performance**:
  - Delivered robust results with high accuracy.
  - Provided a balanced classification report and F1 scores.

#### XGBoost Classifier:
- **Hyperparameters Tuned**:
  - `learning_rate`.
  - `n_estimators`.
- **Performance**:
  - High precision, recall, and overall accuracy.
  - Performed well in detecting DDoS attacks.

**Results Summary**: Consolidated metrics including accuracy, confusion matrices, and classification reports for all models to compare performance.


### Deep Learning Model
A neural network was implemented using TensorFlow/Keras:

#### Architecture:
- **Input Layer**: 64 neurons.
- **Hidden Layers**:
  - Two hidden layers with ReLU activation.
  - Batch normalization to stabilize training.
  - Dropout (30%) to prevent overfitting.
- **Output Layer**: Softmax activation for multi-class classification.

#### Optimization:
- Adam optimizer with learning rate adjustments.
- Early stopping and learning rate reduction on validation loss stagnation.

#### Performance:
- Plotted training/validation accuracy and loss curves.
- Generated confusion matrix and classification report.
- Delivered competitive results with strong accuracy and recall.


## Results and Findings
---
### Machine Learning Models:
- SVM, Random Forest, and XGBoost achieved high accuracy and F1 scores.
- Hyperparameter tuning significantly enhanced performance.

### Deep Learning Model:
- Achieved competitive accuracy and recall.
- Visualization of accuracy/loss curves highlighted model performance.

### EDA Insights:
- Imbalanced class distribution observed and addressed during evaluation.
- Outliers detected in numeric features through boxplots.

### Feature Engineering:
- Top features contributed significantly to classification accuracy, improving model predictions.


## Usage
---
### Steps:
1. **Clone the Repository**  
    ```bash
    $ git clone https://github.com/saghal/CIC-DDoS2019-ML-Detection.git
    $ cd CIC-DDoS2019-ML-Detection
    ```
2. **Install Dependencies**  
Ensure all required libraries are installed as specified in `requirements.txt`.  
    ```bash
    $ pip install -r requirements.txt
    ```
3. **Run the Pipeline**  
- For Jupyter Notebooks:  
  Open and run the notebook.  
  ```  
  jupyter notebook  
  ```  
- Or execute the script directly:  
  ```  
  python anomaly_detection.py  
  ```  

### Outputs:
- EDA visualizations and reports.
- Feature importance plots.
- Model performance metrics and confusion matrices.






