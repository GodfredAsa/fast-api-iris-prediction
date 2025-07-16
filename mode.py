import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib

# 1. Load the Iris Dataset
def load_data():
    columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    df = pd.read_csv('iris.csv', names=columns)
    df = df.dropna()  # Remove any empty rows
    return df

def feature_engineering(df):
    # Add a domain-knowledge feature: petal_length / petal_width
    df['petal_ratio'] = df['petal_length'] / df['petal_width']
    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'petal_ratio']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    X = pd.DataFrame(X_scaled, columns=features)
    y = df['species']
    return X, y

def feature_selection(X, y):
    # Use ANOVA F-value to select the top 2 features
    selector = SelectKBest(score_func=f_classif, k=2)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    print('Selected features:', list(selected_features))
    # Save the selector for use in the API
    joblib.dump(selector, 'selector.pkl')
    return X_selected

def model_selection(X_selected, y):
    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=200, random_state=42)
    }
    for name, model in models.items():
        scores = cross_val_score(model, X_selected, y, cv=5)
        print(f'{name}: Mean accuracy = {scores.mean():.3f} (+/- {scores.std():.3f})')

def train_and_save(X_selected, y):
    final_model = RandomForestClassifier(random_state=42)
    final_model.fit(X_selected, y)
    joblib.dump(final_model, 'iris_model.pkl')
    print('Model saved as iris_model.pkl')

def main():
    df = load_data()
    X, y = feature_engineering(df)
    X_selected = feature_selection(X, y)
    model_selection(X_selected, y)
    train_and_save(X_selected, y)

if __name__ == "__main__":
    main() 