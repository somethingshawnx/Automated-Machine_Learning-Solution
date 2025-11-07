import streamlit as st
import pandas as pd
# --- THIS IS THE FIX ---
from src.utils.insights import configure_groq, get_data_overview, get_model_insights
# --- END OF FIX ---

def show_insights_page():
    """Renders the AI-powered insights tab."""
    st.header("AI-Powered Data Insights (Groq)")
    
    # --- 1. Configure Groq ---
    client = configure_groq()
    if client is None:
        return # Stop execution if API is not configured

    # --- 2. Data Overview ---
    st.subheader("1. Get Data Overview")
    if st.button("Analyze Raw Data"):
        if st.session_state['data'] is not None:
            with st.spinner("Groq is analyzing your data... (this will be fast!)"):
                df = st.session_state['data']
                overview = get_data_overview(client, df)
                st.markdown(overview)
        else:
            st.warning("Please upload data first.")
            
    # --- 3. Model Performance Insights ---
    st.subheader("2. Get Model Performance Insights")
    if st.button("Analyze Model Results"):
        if 'model_results' in st.session_state and st.session_state['model_results'] is not None:
            with st.spinner("Groq is analyzing your model results... (this will be fast!)"):
                results_df = st.session_state['model_results']
                trained_models = st.session_state['trained_models']
                X_test = st.session_state['X_test']
                
                # Get best model
                best_model_name = results_df.iloc[0]['Model']
                best_model = trained_models[best_model_name]
                
                # Get feature importances if they exist
                if hasattr(best_model, 'feature_importances_'):
                    imp = pd.DataFrame({
                        'Feature': X_test.columns,
                        'Importance': best_model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                else:
                    imp = pd.DataFrame(columns=["Feature", "Importance"])
                    st.info("The best model does not have feature importances.")
                
                insights = get_model_insights(client, results_df, best_model_name, imp)
                st.markdown(insights)
        else:
            st.warning("Please train models first to get insights.")
