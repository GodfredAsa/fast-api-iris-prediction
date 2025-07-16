# Mastering Machine Learning: A Practical Guide with the Iris Dataset

## Table of Contents
1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Feature Engineering](#feature-engineering)
4. [Feature Selection](#feature-selection)
5. [Model Selection](#model-selection)
6. [Building an API for Predictions](#building-an-api-for-predictions)
7. [References](#references)

---

## Introduction
This guide is designed to help you master machine learning by focusing on the most critical aspects: **feature engineering**, **feature selection**, and **model selection**. We use the classic [Iris dataset](https://archive.ics.uci.edu/ml/datasets/iris) as a running example, and demonstrate how to build an API for making predictions.

## Getting Started

### 1. Download the Iris Dataset
The Iris dataset can be downloaded from the UCI repository:
```bash
curl -o iris.csv https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
```

### 2. Set Up the Environment
Create a virtual environment and install requirements:
```bash
python3 -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

#### requirements.txt
```
pandas
scikit-learn
joblib
jupyter
fastapi
uvicorn
```

### 3. Read the Dataset in Jupyter Notebook
Start Jupyter Notebook:
```bash
jupyter notebook
```

Read the dataset in a notebook cell:
```python
import pandas as pd

# The dataset has no headers, so we add them manually
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
df = pd.read_csv('iris.csv', names=columns)
df.head()
```

---

## Feature Engineering
Feature engineering is the process of transforming raw data into features that better represent the underlying problem to the predictive models.

- **Scaling:** Standardize features to have zero mean and unit variance.
- **Polynomial Features:** Sometimes, interactions or polynomial terms improve model performance.
- **Domain Knowledge:** For Iris, ratios like `petal_length/petal_width` can be informative.

**Example:**
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df.iloc[:, :-1]), columns=columns[:-1])
df_scaled['species'] = df['species']
df_scaled.head()
```

## Feature Selection
Feature selection helps improve model performance, reduce overfitting, and enhance interpretability.

- **Correlation Analysis:** Remove highly correlated (redundant) features.
- **Univariate Statistics:** Use ANOVA, chi-squared, or mutual information to select features.
- **Model-Based Selection:** Use feature importances from models like Random Forests.

**Why select or drop a feature?**
- If a feature is highly correlated with another, it may be redundant.
- If a feature has low importance or low statistical association with the target, it may be dropped.

**Example:**
```python
from sklearn.feature_selection import SelectKBest, f_classif
X = df.iloc[:, :-1]
y = df['species']
selector = SelectKBest(score_func=f_classif, k=2)
X_new = selector.fit_transform(X, y)
print('Selected features:', X.columns[selector.get_support()])
```

## Model Selection
Model selection involves choosing the best algorithm for your data and problem.

- **Try Multiple Models:** Logistic Regression, SVM, Random Forest, etc.
- **Cross-Validation:** Use k-fold cross-validation to estimate model performance.
- **Statistical Backing:** Compare models using metrics like accuracy, precision, recall, F1-score, and statistical tests (e.g., paired t-test).

**Example:**
```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

models = {
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression(max_iter=200)
}
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5)
    print(f'{name}: Mean accuracy = {scores.mean():.3f} (+/- {scores.std():.3f})')
```

**Why select a model?**
- Choose the model with the best cross-validated performance and interpretability for your use case.
- Use statistical tests to confirm if the difference in performance is significant.

## Building an API for Predictions
Once you have a trained model, you can deploy it as an API using FastAPI.

### Running the API

1. Make sure you have run the training script (e.g., `mode.py`) to generate `iris_model.pkl`.
2. Start the API server:
   ```bash
   uvicorn api:app --reload
   ```

### API Endpoints

#### GET /data
Returns a JSON example of the expected data format for predictions:
```json
{
  "expected_format": [
    {
      "sepal_length": 5.1,
      "sepal_width": 3.5,
      "petal_length": 1.4,
      "petal_width": 0.2,
      "petal_ratio": 7.0
    },
    {
      "sepal_length": 6.2,
      "sepal_width": 2.8,
      "petal_length": 4.8,
      "petal_width": 1.8,
      "petal_ratio": 2.67
    }
  ]
}
```

#### POST /predict
Accepts a CSV file upload (with columns: sepal_length, sepal_width, petal_length, petal_width; petal_ratio is auto-calculated if missing) and returns predictions for each row.

### 🌸 Iris Dataset Sample format 

| Sepal Length | Sepal Width | Petal Length | Petal Width |
|--------------|-------------|--------------|-------------|
| 5.1          | 3.5         | 1.4          | 0.2         |
| 4.9          | 3.0         | 1.4          | 0.2         |
| 4.4          | 2.9         | 1.4          | 0.2         |
| 4.9          | 3.1         | 1.5          | 0.1         |
| 5.4          | 3.7         | 1.5          | 0.2         |
| 4.8          | 3.4         | 1.6          | 0.2         |
| 4.8          | 3.0         | 1.4          | 0.1         |
| 4.3          | 3.0         | 1.1          | 0.1         |

**Example using curl:**
```bash
curl -X POST "http://localhost:8000/predict" -F "file=@iris.csv"
```

**Response:**
```json
{
  "predictions": ["Iris-setosa", "Iris-setosa", ...]
}
```

## References
- [UCI Machine Learning Repository: Iris Data Set](https://archive.ics.uci.edu/ml/datasets/iris)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)