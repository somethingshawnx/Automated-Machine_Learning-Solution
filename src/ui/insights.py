from groq import Groq
import os
import streamlit as st
from dotenv import load_dotenv

def configure_groq():
    """Configures the Groq API. Tries .env for local, then Streamlit secrets for deployment."""
    
    # 1. Try to load key from .env file (for local development)
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    
    # 2. If not found in .env, try Streamlit secrets (for deployment)
    if not api_key:
        try:
            # Note: st.secrets works automatically on Streamlit Cloud
            api_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            # If still not found, show error message
            st.error("GROQ_API_KEY not found.")
            st.info("Please create a **.env** file in your root directory and add: GROQ_API_KEY='your_key_here'")
            return None
    
    if not api_key:
        st.error("GROQ_API_KEY is not set.")
        return None
        
    try:
        # Initialize Groq Client without any extra arguments (this is the key fix)
        client = Groq(api_key=api_key)
        return client
    except Exception as e:
        # This will catch initialization errors, but should now ignore proxies
        st.error(f"Error configuring Groq: {e}")
        return None

def get_data_overview(client, df):
    """Generates a data overview using Groq."""
    
    data_head = df.head().to_string()
    data_describe = df.describe().to_string()
    
    prompt = f"""
    You are a helpful data analyst. I have a dataset.
    Here is the output of df.head():
    {data_head}

    Here is the output of df.describe():
    {data_describe}

    Please provide a brief, high-level overview of this data.
    - What kind of data does it seem to be?
    - Are there any obvious data quality issues or interesting patterns (e.g., high variance, potential outliers)?
    - What are the key features?
    """
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful data analyst."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.1-8b-instant", 
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error generating insights: {e}"

def get_model_insights(client, results_df, best_model_name, feature_importances):
    """Generates model insights using Groq."""
    
    results_string = results_df.to_string()
    
    prompt = f"""
    You are a machine learning expert. I have trained several models and have the following results:
    
    Model Performance Table:
    {results_string}
    
    The best performing model is: {best_model_name}
    
    Here are the top 10 feature importances for the best model:
    {feature_importances.head(10).to_string()}
    
    Please provide a concise analysis:
    1.  Explain in simple terms what the performance table means. Which model won and why?
    2.  Based on the feature importances, what are the most important factors driving the predictions?
    3.  Give one simple, actionable suggestion for how I could potentially improve the model (e.g., "You might want to investigate 'feature_x' more").
    """
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a machine learning expert."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.1-8b-instant",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error generating insights: {e}"
