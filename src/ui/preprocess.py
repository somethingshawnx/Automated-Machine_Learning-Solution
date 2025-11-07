import streamlit as st
import pandas as pd
# --- THIS IS THE FIX ---
from src.preprocessing.core import (
# --- END OF FIX ---
    handle_missing_values, 
    encode_categorical_features, 
    scale_numeric_features
)

def show_preprocess_page(df: pd.DataFrame):
    """
    Renders the data preprocessing tab.
    """
    
    st.header("Clean and Preprocess Data")
    st.write("First, select your target variable. Then, apply transformations.")
    
    # --- 1. SELECT TARGET VARIABLE ---
    st.subheader("1. Select Target Variable (Y)")
    all_columns = df.columns.tolist()
    
    default_target_index = 0
    if 'target_variable' in st.session_state and st.session_state['target_variable'] is not None:
        try:
            default_target_index = all_columns.index(st.session_state['target_variable'])
        except ValueError:
            pass # Target not in columns, use default
            
    target_variable = st.selectbox(
        "Which column are you trying to predict?", 
        all_columns, 
        index=default_target_index
    )
    # Store it in session state immediately
    st.session_state['target_variable'] = target_variable
    
    # --- 2. Preprocessing Options ---
    st.subheader("2. Preprocessing Options")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Missing Values (Numeric)**")
        num_impute_strategy = st.selectbox(
            "Select strategy:",
            ('mean', 'median'), key='num_impute'
        )
        st.markdown("**Missing Values (Categorical)**")
        cat_impute_strategy = st.selectbox(
            "Select strategy:",
            ('most_frequent', 'constant'), key='cat_impute'
        )

    with col2:
        st.markdown("**Categorical Encoding**")
        encoding_strategy = st.selectbox(
            "Select strategy:",
            ('one_hot', 'label'), key='encode'
        )
        st.markdown("**Feature Scaling**")
        scaling_strategy = st.selectbox(
            "Select scaling strategy:",
            ('standard', 'min_max', 'none'), key='scale'
        )

    # --- 3. Apply Button ---
    st.subheader("3. Apply Preprocessing")
    if st.button("Apply Preprocessing", type="primary"):
        with st.spinner("Processing data..."):
            try:
                df_processed = df.copy()
                
                # Step 1: Handle Missing Values
                df_processed = handle_missing_values(
                    df_processed, num_impute_strategy, cat_impute_strategy
                )
                
                # Step 2: Encode Categorical Features
                df_processed = encode_categorical_features(
                    df_processed, target_variable, encoding_strategy
                )
                
                # Step 3: Scale Numeric Features
                if scaling_strategy != 'none':
                    df_processed = scale_numeric_features(
                        df_processed, target_variable, scaling_strategy
                    )
                
                st.session_state['processed_data'] = df_processed
                st.success("Preprocessing applied successfully!")
                
            except Exception as e:
                st.error(f"An error occurred during preprocessing: {e}")

    # --- Show Processed Data ---
    if 'processed_data' in st.session_state and st.session_state['processed_data'] is not None:
        with st.expander("Preview of Processed Data"):
            st.dataframe(st.session_state['processed_data'].head())
    else:
        st.info("Processed data will appear here after you apply transformations.")
