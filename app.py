import streamlit as st
import pandas as pd
import warnings

from streamlit_option_menu import option_menu

# --- Import our UI modules ---
from src.ui.explore import show_explore_page
from src.ui.preprocess import show_preprocess_page
from src.ui.train import show_train_page
from src.ui.evaluate import show_evaluate_page
from src.ui.insights import show_insights_page
from src.ui.chatbot import show_chatbot_page  # NEW IMPORT

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Machine Learnning Studio",
    page_icon="🤖",
    layout="wide"
)

def load_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file '{file_name}' not found. Using default styles.")

load_css('style.css')

# --- Initialize Session State ---
if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'processed_data' not in st.session_state:
    st.session_state['processed_data'] = None
if 'target_variable' not in st.session_state:
    st.session_state['target_variable'] = None
if 'model_results' not in st.session_state:
    st.session_state['model_results'] = None
if 'trained_models' not in st.session_state:
    st.session_state['trained_models'] = None
if 'X_test' not in st.session_state:
    st.session_state['X_test'] = None
if 'y_test' not in st.session_state:
    st.session_state['y_test'] = None
if 'uploaded_file_name' not in st.session_state:
    st.session_state['uploaded_file_name'] = None
if 'problem_type' not in st.session_state:
    st.session_state['problem_type'] = None
if 'X_train' not in st.session_state:
    st.session_state['X_train'] = None
if 'y_train' not in st.session_state:
    st.session_state['y_train'] = None
if 'tuned_model_results' not in st.session_state:
    st.session_state['tuned_model_results'] = None
if 'messages' not in st.session_state:  # NEW: For chatbot
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you today?"}]

st.title("🤖 AutoML Workbench")
st.write("Upload, explore, preprocess, train, and evaluate models—all in one place.")

selected_tab = option_menu(
    menu_title=None,
    # --- ADDED "Chatbot" ---
    options=["Upload", "Explore", "Preprocess", "Train", "Evaluate", "Insights", "Chatbot"],
    icons=["cloud-upload", "bar-chart-line", "gear", "robot", "clipboard-data", "lightbulb", "chat-dots"],
    # --- END OF CHANGE ---
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#0E1117"},
        "icon": {"color": "white", "font-size": "18px"},
        "nav-link": {
            "font-size": "16px",
            "text-align": "left",
            "margin": "0px",
            "--hover-color": "#2c3443",
        },
        "nav-link-selected": {"background-color": "#4CAF50"},
    },
)

if selected_tab == "Upload":
    st.header("1. Upload Your Dataset")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        if uploaded_file.name != st.session_state['uploaded_file_name']:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state['data'] = df
                st.session_state['uploaded_file_name'] = uploaded_file.name
                
                # Reset all downstream states
                st.session_state['processed_data'] = None 
                st.session_state['target_variable'] = None
                st.session_state['model_results'] = None
                st.session_state['trained_models'] = None
                st.session_state['X_test'] = None
                st.session_state['y_test'] = None
                st.session_state['problem_type'] = None
                st.session_state['X_train'] = None
                st.session_state['y_train'] = None
                st.session_state['tuned_model_results'] = None
                st.session_state['messages'] = [{"role": "assistant", "content": "How can I help you today?"}] # Reset chat
                
                st.success(f"File '{uploaded_file.name}' uploaded successfully!")
                
            except Exception as e:
                st.error(f"Error reading file: {e}")
        
        if st.session_state['data'] is not None:
            st.subheader("Data Preview (First 5 Rows)")
            st.dataframe(st.session_state['data'].head())
            
    else:
        if st.session_state['uploaded_file_name'] is not None:
            # Clear all session data if file is removed
            st.session_state['data'] = None
            st.session_state['uploaded_file_name'] = None
            st.session_state['processed_data'] = None 
            st.session_state['target_variable'] = None
            st.session_state['model_results'] = None
            st.session_state['trained_models'] = None
            st.session_state['X_test'] = None
            st.session_state['y_test'] = None
            st.session_state['problem_type'] = None
            st.session_state['X_train'] = None
            st.session_state['y_train'] = None
            st.session_state['tuned_model_results'] = None
            st.session_state['messages'] = [{"role": "assistant", "content": "How can I help you today?"}]
            st.rerun() 
            
        st.info("Please upload a CSV file to begin.")

elif selected_tab == "Explore":
    if st.session_state['data'] is not None:
        show_explore_page(st.session_state['data'])
    else:
        st.warning("Please upload data in the 'Upload' tab first.")

elif selected_tab == "Preprocess":
    if st.session_state['data'] is not None:
        show_preprocess_page(st.session_state['data'])
    else:
        st.warning("Please upload data in the 'Upload' tab first.")

elif selected_tab == "Train":
    if st.session_state['processed_data'] is not None:
        show_train_page(st.session_state['processed_data'])
    elif st.session_state['data'] is not None:
        st.warning("Please select your target and apply transformations in the 'Preprocess' tab first.")
    else:
        st.warning("Please upload data in the 'Upload' tab first.")

elif selected_tab == "Evaluate":
    if st.session_state['model_results'] is not None:
        show_evaluate_page()
    elif st.session_state['processed_data'] is not None:
        st.warning("Please train your models in the 'Train' tab first.")
    else:
        st.warning("Please upload and preprocess your data first.")

elif selected_tab == "Insights":
    if st.session_state['data'] is not None:
        show_insights_page()
    else:
        st.warning("Please upload data in the 'Upload' tab first.")

# --- NEW: Chatbot Page ---
elif selected_tab == "Chatbot":

    show_chatbot_page()
