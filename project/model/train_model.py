import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib


def load_data(path='dataset/student_data.csv'):
    """Load dataset from CSV file. If the file does not exist, generate a
    synthetic dataset for demonstration purposes.
    """
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        print(f"Dataset not found at {path}, generating synthetic data...")
        rng = np.random.RandomState(42)
        n = 500
        df = pd.DataFrame({
            'grades': rng.randint(40, 100, size=n),
            'gpa': np.round(rng.uniform(2.0, 4.0, size=n), 2),
            'attendance': np.round(rng.uniform(50, 100, size=n), 1),
            'behavior': rng.choice(['good', 'average', 'poor'], size=n),
            'socio_eco': rng.choice(['low', 'medium', 'high'], size=n),
        })
        # create a simple rule for risk label
        def risk_row(r):
            score = r['grades'] + r['attendance'] + (0 if r['behavior']=='poor' else 50)
            if score < 180:
                return 'High'
            elif score < 240:
                return 'Medium'
            else:
                return 'Low'
        df['risk'] = df.apply(risk_row, axis=1)
        # save so user can inspect
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
    return df


def preprocess(df):
    """Handle missing values, encode categorical variables, and prepare
    features and labels for training.
    """
    # drop rows with missing target
    df = df.dropna(subset=['risk'])

    # features and label
    X = df.drop(columns=['risk'])
    y = df['risk']

    # label encode the target
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    return X, y_enc, le


def train_and_select_model(df):
    """Train multiple classifiers and choose the best-performing one based on
    F1-score. Save the selected model pipeline (including preprocessing) to
    disk.
    """
    X, y, label_encoder = preprocess(df)

    # split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # define which columns are numeric/categorical
    numeric_features = ['grades', 'gpa', 'attendance']
    categorical_features = ['behavior', 'socio_eco']

    # create preprocessing pipeline
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
        ]
    )

    # candidate models
    candidates = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'RandomForest': RandomForestClassifier(n_estimators=100),
        'DecisionTree': DecisionTreeClassifier(),
        'SVM': SVC(probability=True),
    }

    best_model = None
    best_score = -np.inf
    metrics = {}

    for name, estimator in candidates.items():
        print(f"Training {name}...")
        clf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', estimator)])
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

        metrics[name] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1
        }

        print(f"{name} -> acc: {acc:.3f}, prec: {prec:.3f}, rec: {rec:.3f}, f1: {f1:.3f}")

        if f1 > best_score:
            best_score = f1
            best_model = clf
            best_name = name

    print("\nModel comparison:\n")
    for name, m in metrics.items():
        print(f"{name}: {m}")

    print(f"\nBest model -> {best_name} with F1={best_score:.3f}")

    # save the pipeline and label encoder together
    os.makedirs('model', exist_ok=True)
    joblib.dump({'pipeline': best_model, 'label_encoder': label_encoder},
                'model/saved_model.pkl')
    print("Saved best model to model/saved_model.pkl")


if __name__ == '__main__':
    df = load_data()
    train_and_select_model(df)
