import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, auc, mean_squared_error, r2_score, accuracy_score
import numpy as np
import pickle
from io import BytesIO

# --- IMPORTS FOR TUNING ---
from src.training.tune import get_param_grids, tune_model
from src.training.train import get_classification_models, get_regression_models

# --- PLOTTING FUNCTIONS ---
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm)
    fig = px.imshow(cm_df, text_auto=True, 
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    title="Confusion Matrix")
    fig.update_layout(paper_bgcolor="#0E1117", plot_bgcolor="#0E1117", font_color="white")
    fig.update_xaxes(side="top")
    st.plotly_chart(fig, width='stretch')

def plot_roc_curve(y_true, y_proba):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                             name=f'ROC curve (area = {roc_auc:.2f})'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                             line=dict(dash='dash', color='gray'), name='Chance'))
    
    fig.update_layout(
        title='Receiver Operating Characteristic (ROC) Curve',
        xaxis_title='False PositiveRate',
        yaxis_title='True Positive Rate',
        paper_bgcolor="#0E1117", plot_bgcolor="#0E1117", font_color="white"
    )
    st.plotly_chart(fig, width='stretch')

def plot_feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        imp_df = imp_df.sort_values(by='Importance', ascending=False).head(20)
        
        fig = px.bar(imp_df, x='Importance', y='Feature', orientation='h',
                     title="Top 20 Feature Importances")
        fig.update_layout(paper_bgcolor="#0E1117", plot_bgcolor="#0E1117", font_color="white")
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("This model does not have a 'feature_importances_' attribute.")

def plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    fig = px.scatter(x=y_pred, y=residuals, title="Residual Plot",
                     labels={'x': 'Predicted Values', 'y': 'Residuals'})
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    fig.update_layout(paper_bgcolor="#0E1117", plot_bgcolor="#0E1117", font_color="white")
    st.plotly_chart(fig, width='stretch')

# --- METRIC CARD FUNCTION ---
def display_metric_card(title, value, key):
    st.markdown(f"""
    <h3>{title}</h3>
    <p style="font-size: 2.5em; font-weight: bold; margin-top: -10px;">{value}</p>
    """, unsafe_allow_html=True)

# --- CUSTOM ACCURACY FUNCTION ---
def calculate_custom_accuracy(y_true, y_pred, tolerance=0.1):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    epsilon = 1e-6
    abs_percentage_error = np.abs((y_true - y_pred) / (y_true + epsilon))
    correct_within_tolerance = np.sum(abs_percentage_error <= tolerance)
    return (correct_within_tolerance / len(y_true))

# --- MAIN PAGE FUNCTION ---
def show_evaluate_page():
    st.header("Evaluate Model Performance")
    
    if 'trained_models' not in st.session_state or not st.session_state['trained_models']:
        st.warning("Please train models in the 'Train Models' tab first.")
        return

    # Load data from session state
    trained_models = st.session_state['trained_models']
    model_results = st.session_state['model_results']
    X_train = st.session_state['X_train']
    y_train = st.session_state['y_train']
    X_test = st.session_state['X_test']
    y_test = st.session_state['y_test']
    problem_type = st.session_state['problem_type']
    
    # --- THIS IS THE NEW PART ---
    st.subheader("Model Comparison")
    with st.expander("Show Full Model Comparison Table (from 'Train' tab)"):
        st.dataframe(model_results)
    # --- END OF NEW PART ---

    st.subheader("1. Select a Model to Evaluate")
    sorted_model_names = model_results['Model'].tolist()
    
    selected_model_name = st.selectbox(
        "Choose a model:",
        sorted_model_names
    )
    
    if selected_model_name:
        model = trained_models[selected_model_name]
        y_pred = model.predict(X_test)
        
        st.subheader(f"Detailed Evaluation for: {selected_model_name}")

        if problem_type == "Classification":
            col1, col2, col3 = st.columns(3)
            metrics = model_results[model_results['Model'] == selected_model_name].iloc[0]
            with col1:
                display_metric_card("Accuracy", f"{metrics['Accuracy']:.4f}", "acc_card")
            with col2:
                display_metric_card("F1-Score", f"{metrics['F1-Score']:.4f}", "f1_card")
            with col3:
                display_metric_card("ROC-AUC", f"{metrics['ROC-AUC']:.4f}", "roc_card")
            
            plot_col1, plot_col2 = st.columns(2)
            with plot_col1:
                plot_confusion_matrix(y_test, y_pred)
            with plot_col2:
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test)[:, 1]
                    plot_roc_curve(y_test, y_proba)
                else:
                    st.info("This model does not support `predict_proba`.")
            
            plot_feature_importance(model, X_test.columns)

        elif problem_type == "Regression":
            custom_acc = calculate_custom_accuracy(y_test, y_pred, tolerance=0.1)
            
            col1, col2, col3, col4 = st.columns(4)
            metrics = model_results[model_results['Model'] == selected_model_name].iloc[0]
            
            with col1:
                display_metric_card("R-Squared (R²)", f"{metrics['R-Squared (R²)']:.4f}", "r2_card")
            with col2:
                display_metric_card("RMSE", f"{metrics['Root Mean Squared Error (RMSE)']:.4f}", "rmse_card")
            with col3:
                display_metric_card("MAE", f"{metrics['Mean Absolute Error (MAE)']:.4f}", "mae_card")
            with col4:
                display_metric_card("Acc. (within 10%)", f"{custom_acc:.2%}", "custom_acc_card")
            
            plot_col1, plot_col2 = st.columns(2)
            with plot_col1:
                plot_residuals(y_test, y_pred)
            with plot_col2:
                plot_feature_importance(model, X_test.columns)
            
        st.divider()

        # (The rest of the file is unchanged)
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Download Model")
            model_bytes = pickle.dumps(model)
            st.download_button(
                label=f"Download {selected_model_name} (.pkl)",
                data=model_bytes,
                file_name=f"{selected_model_name.lower().replace(' ', '_')}.pkl",
                mime="application/octet-stream"
            )
        
        with col2:
            st.subheader(f"Tune {selected_model_name}")
            class_grids, reg_grids = get_param_grids()
            param_grid = class_grids.get(selected_model_name, {}) if problem_type == "Classification" else reg_grids.get(selected_model_name, {})
            
            if not param_grid:
                st.info(f"No hyperparameter grid defined for {selected_model_name}.")
            else:
                if st.button(f"Tune {selected_model_name} (Fast)", type="primary"):
                    scoring = 'accuracy' if problem_type == "Classification" else 'r2'
                    with st.spinner(f"Tuning {selected_model_name}..."):
                        try:
                            base_model = get_classification_models()[selected_model_name] if problem_type == "Classification" else get_regression_models()[selected_model_name]
                            
                            tuned_model, best_params, best_score = tune_model(
                                base_model, param_grid, X_train, y_train, scoring
                            )
                            
                            tuned_y_pred = tuned_model.predict(X_test)
                            metric_name = "Test Accuracy" if problem_type == "Classification" else "Test R²"
                            tuned_metric = accuracy_score(y_test, tuned_y_pred) if problem_type == "Classification" else r2_score(y_test, tuned_y_pred)

                            st.session_state['tuned_model_results'] = {
                                'name': f"Tuned {selected_model_name}",
                                'model': tuned_model,
                                'best_params': best_params,
                                'cv_score': best_score,
                                'test_metric': tuned_metric,
                                'metric_name': metric_name
                            }
                            st.success("Tuning complete!")
                        
                        except Exception as e:
                            st.error(f"Error during tuning: {e}")

        if 'tuned_model_results' in st.session_state and st.session_state['tuned_model_results']:
            results = st.session_state['tuned_model_results']
            if results['name'] == f"Tuned {selected_model_name}":
                st.subheader("Tuning Results")
                
                t_col1, t_col2 = st.columns(2)
                with t_col1:
                    display_metric_card(f"CV Score ({'accuracy' if problem_type == 'Classification' else 'r2'})", f"{results['cv_score']:.4f}", "tune_cv_card")
                with t_col2:
                    display_metric_card(results['metric_name'], f"{results['test_metric']:.4f}", "tune_test_card")

                with st.expander("Show Best Parameters"):
                    st.json(results['best_params'])
                
                st.subheader("Download Tuned Model")
                tuned_model_bytes = pickle.dumps(results['model'])
                st.download_button(
                    label=f"Download {results['name']} (.pkl)",
                    data=tuned_model_bytes,
                    file_name=f"{results['name'].lower().replace(' ', '_')}.pkl",
                    mime="application/octet-stream"
                )