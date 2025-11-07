import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff

def show_explore_page(df: pd.DataFrame):
    """
    Renders the data exploration tab.
    """
    st.header("Explore Your Data")

    # --- 1. Summary Statistics (in an expander) ---
    with st.expander("📊 Show Summary Statistics"):
        st.write("Get a quick overview of the numerical features in your dataset.")
        st.dataframe(df.describe())

    # --- 2. Data Types (in an expander) ---
    with st.expander("🔡 Show Data Types"):
        st.write("Review the data types for each column.")
        st.dataframe(df.dtypes.astype(str), width='stretch')

    # --- 3. Correlation Heatmap ---
    st.subheader("🔥 Correlation Heatmap")
    st.write("Visualize the correlation between numerical features.")
    
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    if not numeric_df.empty:
        corr = numeric_df.corr()
        fig = px.imshow(corr, 
                        text_auto=True, 
                        aspect="auto", 
                        color_continuous_scale='RdYlBu_r',
                        title="Correlation Matrix")
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("No numerical columns found to calculate correlation.")

    # --- 4. Column Distribution ---
    st.subheader("📈 Column Distributions")
    st.write("Select columns to visualize their distributions.")
    
    all_columns = df.columns.tolist()
    selected_cols = st.multiselect("Select columns:", all_columns, default=all_columns[:3])
    
    if selected_cols:
        for col in selected_cols:
            try:
                if pd.api.types.is_numeric_dtype(df[col]):
                    fig = px.histogram(df, x=col, marginal="box", title=f"Distribution of {col}")
                    st.plotly_chart(fig, width='stretch')
                else:
                    st.write(f"**Value Counts for {col}**")
                    counts = df[col].value_counts().reset_index()
                    counts.columns = [col, 'count']
                    
                    if len(counts) > 20:
                        counts = counts.head(20)
                        st.warning(f"Displaying top 20 categories for {col}.")

                    fig = px.bar(counts, x=col, y='count', title=f"Value Counts for {col}")
                    st.plotly_chart(fig, width='stretch')

            except Exception as e:
                st.error(f"Could not plot {col}: {e}")
    else:
        st.info("Select one or more columns to see their distributions.")