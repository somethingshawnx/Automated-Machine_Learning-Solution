import numpy as np
from sklearn.model_selection import RandomizedSearchCV

def get_param_grids():
    """
    Returns hyperparameter grids for RandomizedSearchCV.
    Using distributions (like loguniform) is often better, 
    but for simplicity in the app, we'll use lists of values.
    """
    
    # --- Classification Grids ---
    classification_grids = {
        "Logistic Regression": {
            'C': [0.01, 0.1, 1, 10, 100],
            'solver': ['liblinear', 'saga']
        },
        "Random Forest": {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        "XGBoost": {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.7, 0.8, 0.9]
        },
        "LightGBM": {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'num_leaves': [20, 30, 40]
        },
        "Support Vector Machine (SVC)": {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto', 0.1, 1],
            'kernel': ['rbf', 'poly']
        },
        "Decision Tree": {
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'criterion': ['gini', 'entropy']
        },
        "K-Nearest Neighbors (KNN)": {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'metric': ['minkowski', 'euclidean']
        }
        # Add other models as needed
    }

    # --- Regression Grids ---
    regression_grids = {
        "Linear Regression": {}, # Linear Regression has no real hyperparameters to tune
        "Random Forest": {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        "XGBoost": {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.7, 0.8, 0.9]
        },
        "LightGBM": {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'num_leaves': [20, 30, 40]
        },
        "Support Vector Machine (SVR)": {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto', 0.1, 1],
            'kernel': ['rbf', 'poly']
        },
        "Decision Tree": {
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'criterion': ['squared_error', 'absolute_error']
        },
        "K-Nearest Neighbors (KNN)": {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'metric': ['minkowski', 'euclidean']
        },
        "Ridge Regression": {
            'alpha': [0.1, 1.0, 10.0, 100.0]
        },
        "Lasso Regression": {
            'alpha': [0.01, 0.1, 1.0, 10.0]
        }
        # Add other models as needed
    }
    
    return classification_grids, regression_grids


def tune_model(model, param_grid, X_train, y_train, scoring, n_iter=10):
    """
    Performs RandomizedSearchCV on a given model.
    """
    if not param_grid:
        return model, {}, 0 # No tuning possible
        
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=n_iter,  # Number of parameter settings that are sampled
        cv=3,           # 3-fold cross-validation
        verbose=0,
        random_state=42,
        n_jobs=-1,      # Use all available cores
        scoring=scoring
    )
    
    random_search.fit(X_train, y_train)
    
    return random_search.best_estimator_, random_search.best_params_, random_search.best_score_