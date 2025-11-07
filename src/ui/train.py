<<<<<<< HEAD
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from src.training.train import (
    get_classification_models, get_regression_models,
    train_models, evaluate_classification, evaluate_regression
)
import re

def show_train_page(df: pd.DataFrame):
    """
    Renders the model training tab.
    """
    st.header("Train Machine Learning Models")
    
    # --- 1. LOAD TARGET VARIABLE (IT'S ALREADY IN SESSION STATE) ---
    if 'target_variable' not in st.session_state or st.session_state['target_variable'] is None:
        st.warning("Please select a target variable in the 'Preprocess' tab.")
        return
        
    target_variable = st.session_state['target_variable']
    
    # Sanitize column names
    df.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', col) for col in df.columns]
    
    # Make sure target_variable is also sanitized
    target_variable = re.sub(r'[^A-Za-z0-9_]+', '_', target_variable)
    
    # Check if sanitized target is in the columns
    if target_variable not in df.columns:
        st.error(f"Target variable '{target_variable}' not found in processed data columns.")
        return

    st.success(f"Target Variable: **{target_variable}**")

    # 2. --- Select Problem Type ---
    st.subheader("2. Select Problem Type")
    
    if pd.api.types.is_numeric_dtype(df[target_variable]):
        # Check if it's *really* regression or just encoded classification
        if df[target_variable].nunique() < 25: # Arbitrary threshold for classes
            default_problem_type = "Classification"
        else:
            default_problem_type = "Regression"
    else:
        default_problem_type = "Classification"
        
    problem_type = st.radio(
        "What type of problem is this?",
        ("Classification", "Regression"),
        index=0 if default_problem_type == "Classification" else 1
    )
    
    # 3. --- Select Models ---
    st.subheader("3. Select Models to Train")
    
    if problem_type == "Classification":
        all_models = get_classification_models()
    else:
        all_models = get_regression_models()
    
    selected_model_names = st.multiselect(
        "Choose models:", 
        options=list(all_models.keys()), 
        default=list(all_models.keys())[:3]
    )
    
    models_to_train = {name: all_models[name] for name in selected_model_names}

    # 4. --- Train/Test Split ---
    st.subheader("4. Configure Train/Test Split")
    test_size = st.slider("Test Set Size", 0.1, 0.5, 0.2, 0.05)
    
    # 5. --- Start Training ---
    if st.button("Start Training", type="primary", disabled=(not models_to_train)):
        
        if not models_to_train:
            st.warning("Please select at least one model to train.")
            return

        with st.spinner("Training models... This may take a moment."):
            try:
                X = df.drop(columns=[target_variable])
                y = df[target_variable]
                
                # This logic is still needed for classification
                if problem_type == "Classification" and not pd.api.types.is_numeric_dtype(y):
                    y = y.astype('category').cat.codes
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
                
                st.session_state['trained_models'] = train_models(X_train, y_train, models_to_train)
                
                if problem_type == "Classification":
                    results_df = evaluate_classification(st.session_state['trained_models'], X_test, y_test)
                else:
                    results_df = evaluate_regression(st.session_state['trained_models'], X_test, y_test)
                
                st.session_state['model_results'] = results_df
                st.session_state['X_train'] = X_train
                st.session_state['y_train'] = y_train
                st.session_state['X_test'] = X_test
                st.session_state['y_test'] = y_test
                st.session_state['problem_type'] = problem_type
                st.session_state['tuned_model_results'] = None
                
                st.success("Model training and evaluation complete!")
                
            except Exception as e:
                st.error(f"An error occurred during training: {e}")
                
    if 'model_results' in st.session_state and st.session_state['model_results'] is not None:
        st.subheader("Model Performance Results")
        st.dataframe(st.session_state['model_results'])
=======
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from src.training.train import (
    get_classification_models, get_regression_models,
    train_models, evaluate_classification, evaluate_regression
)
import re

def show_train_page(df: pd.DataFrame):
    """
    Renders the model training tab.
    """
    st.header("Train Machine Learning Models")
    
    # --- 1. LOAD TARGET VARIABLE (IT'S ALREADY IN SESSION STATE) ---
    if 'target_variable' not in st.session_state or st.session_state['target_variable'] is None:
        st.warning("Please select a target variable in the 'Preprocess' tab.")
        return
        
    target_variable = st.session_state['target_variable']
    
    # Sanitize column names
    df.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', col) for col in df.columns]
    
    # Make sure target_variable is also sanitized
    target_variable = re.sub(r'[^A-Za-z0-9_]+', '_', target_variable)
    
    # Check if sanitized target is in the columns
    if target_variable not in df.columns:
        st.error(f"Target variable '{target_variable}' not found in processed data columns.")
        return

    st.success(f"Target Variable: **{target_variable}**")

    # 2. --- Select Problem Type ---
    st.subheader("2. Select Problem Type")
    
    if pd.api.types.is_numeric_dtype(df[target_variable]):
        # Check if it's *really* regression or just encoded classification
        if df[target_variable].nunique() < 25: # Arbitrary threshold for classes
            default_problem_type = "Classification"
        else:
            default_problem_type = "Regression"
    else:
        default_problem_type = "Classification"
        
    problem_type = st.radio(
        "What type of problem is this?",
        ("Classification", "Regression"),
        index=0 if default_problem_type == "Classification" else 1
    )
    
    # 3. --- Select Models ---
    st.subheader("3. Select Models to Train")
    
    if problem_type == "Classification":
        all_models = get_classification_models()
    else:
        all_models = get_regression_models()
    
    selected_model_names = st.multiselect(
        "Choose models:", 
        options=list(all_models.keys()), 
        default=list(all_models.keys())[:3]
    )
    
    models_to_train = {name: all_models[name] for name in selected_model_names}

    # 4. --- Train/Test Split ---
    st.subheader("4. Configure Train/Test Split")
    test_size = st.slider("Test Set Size", 0.1, 0.5, 0.2, 0.05)
    
    # 5. --- Start Training ---
    if st.button("Start Training", type="primary", disabled=(not models_to_train)):
        
        if not models_to_train:
            st.warning("Please select at least one model to train.")
            return

        with st.spinner("Training models... This may take a moment."):
            try:
                X = df.drop(columns=[target_variable])
                y = df[target_variable]
                
                # This logic is still needed for classification
                if problem_type == "Classification" and not pd.api.types.is_numeric_dtype(y):
                    y = y.astype('category').cat.codes
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
                
                st.session_state['trained_models'] = train_models(X_train, y_train, models_to_train)
                
                if problem_type == "Classification":
                    results_df = evaluate_classification(st.session_state['trained_models'], X_test, y_test)
                else:
                    results_df = evaluate_regression(st.session_state['trained_models'], X_test, y_test)
                
                st.session_state['model_results'] = results_df
                st.session_state['X_train'] = X_train
                st.session_state['y_train'] = y_train
                st.session_state['X_test'] = X_test
                st.session_state['y_test'] = y_test
                st.session_state['problem_type'] = problem_type
                st.session_state['tuned_model_results'] = None
                
                st.success("Model training and evaluation complete!")
                
            except Exception as e:
                st.error(f"An error occurred during training: {e}")
                
    if 'model_results' in st.session_state and st.session_state['model_results'] is not None:
        st.subheader("Model Performance Results")
        st.dataframe(st.session_state['model_results'])
>>>>>>> 03a601610e7fe25393d778fa7d5705df71ad9e68
        st.info("The best performing models are at the top. You can now analyze them in the 'Evaluate' tab.")