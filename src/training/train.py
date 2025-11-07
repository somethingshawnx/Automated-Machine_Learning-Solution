import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, r2_score, mean_absolute_error
)
import numpy as np

# --- Model Imports ---
# Classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# --- Model Definitions ---

def get_classification_models():
    """Returns a dictionary of classification models."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        "LightGBM": LGBMClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        "Support Vector Machine (SVC)": SVC(probability=True),
        "Decision Tree": DecisionTreeClassifier(),
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(),
        "Gaussian Naive Bayes": GaussianNB(),
        "Linear Discriminant Analysis (LDA)": LinearDiscriminantAnalysis(),
        "Quadratic Discriminant Analysis (QDA)": QuadraticDiscriminantAnalysis(),
    }

def get_regression_models():
    """Returns a dictionary of regression models."""
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(),
        "XGBoost": XGBRegressor(objective='reg:squarederror'),
        "LightGBM": LGBMRegressor(),
        "Support Vector Machine (SVR)": SVR(),
        "Gradient Boosting": GradientBoostingRegressor(),
        "AdaBoost": AdaBoostRegressor(),
        "Decision Tree": DecisionTreeRegressor(),
        "K-Nearest Neighbors (KNN)": KNeighborsRegressor(),
        "Ridge Regression": Ridge(),
        "Lasso Regression": Lasso(),
        "ElasticNet": ElasticNet(),
        "Bayesian Ridge": BayesianRidge(),
    }

def train_models(X_train, y_train, models):
    """Trains a dictionary of models."""
    trained_models = {}
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            trained_models[name] = model
        except Exception as e:
            print(f"Error training {name}: {e}")
    return trained_models

def evaluate_classification(trained_models, X_test, y_test):
    """Evaluates classification models."""
    results = []
    for name, model in trained_models.items():
        try:
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            results.append({
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, average='weighted'),
                "Recall": recall_score(y_test, y_pred, average='weighted'),
                "F1-Score": f1_score(y_test, y_pred, average='weighted'),
                "ROC-AUC": roc_auc_score(y_test, y_proba, average='weighted')
            })
        except Exception as e:
            print(f"Error evaluating {name}: {e}")
    
    return pd.DataFrame(results).sort_values(by="F1-Score", ascending=False)

def evaluate_regression(trained_models, X_test, y_test):
    """Evaluates regression models."""
    results = []
    for name, model in trained_models.items():
        try:
            y_pred = model.predict(X_test)
            
            results.append({
                "Model": name,
                "R-Squared (R²)": r2_score(y_test, y_pred),
                "Mean Squared Error (MSE)": mean_squared_error(y_test, y_pred),
                "Root Mean Squared Error (RMSE)": np.sqrt(mean_squared_error(y_test, y_pred)),
                "Mean Absolute Error (MAE)": mean_absolute_error(y_test, y_pred)
            })
        except Exception as e:
            print(f"Error evaluating {name}: {e}")
            
    return pd.DataFrame(results).sort_values(by="R-Squared (R²)", ascending=False)