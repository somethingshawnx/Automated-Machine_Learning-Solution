# 🤖 MachineLearnig Studio

An interactive, end-to-end web application for automating the machine learning workflow. Built with Streamlit, this tool allows you to upload, visualize, preprocess, and train multiple models, then evaluate and tune the best-performing one. It even includes an AI-powered chatbot to help you analyze your results.



## ✨ Features

This application streamlines the entire ML pipeline into a simple, multi-page interface:

* **1. Upload Data:** Easily upload your CSV datasets.
* **2. Explore Data:** Get a comprehensive overview of your data with:
    * Summary statistics (`.describe()`)
    * Data type information
    * Interactive correlation heatmaps
    * Distribution plots for all features
* **3. Preprocess Data:** A full UI for data cleaning and preparation:
    * **Target Selection:** Select your target variable *before* preprocessing to protect it.
    * **Missing Values:** Handle numeric (mean, median) and categorical (most frequent) data.
    * **Categorical Encoding:** Choose between One-Hot or Label Encoding.
    * **Feature Scaling:** Apply `StandardScaler` or `MinMaxScaler` to your features.
* **4. Train Models:** Automatically train and compare over 20+ ML models for both:
    * **Classification:** Logistic Regression, Random Forest, XGBoost, SVC, etc.
    * **Regression:** Linear Regression, Random Forest, XGBoost, SVR, etc.
* **5. Evaluate Models:** Dive deep into model performance with a visual dashboard:
    * **Comparison Table:** See all models ranked by performance.
    * **Metric Cards:** View key metrics (Accuracy, F1, R², RMSE, etc.) in a clean UI.
    * **Plots:** Generates Confusion Matrices, ROC-AUC Curves, Feature Importance, and Residual Plots.
    * **Model Download:** Download any trained model as a `.pkl` file.
* **6. Hyperparameter Tuning:**
    * Select your best model and run `RandomizedSearchCV` with a single click.
    * See the best parameters and the improved score.
    * Download the new, optimized model.
* **7. AI Insights (Groq):**
    * Get a natural language summary of your raw data.
    * Get an AI-powered analysis of your model results and feature importances.
* **8. AI Chatbot (Groq):**
    * A fully conversational chatbot that is **context-aware**.
    * Ask it questions about your data (`"what's the mean of my 'price' column?"`) or your results (`"what was the best model?"`).

## 🛠 Tech Stack

* **Core:** Python
* **Web App:** Streamlit
* **UI Components:** streamlit-option-menu, `style.css`
* **Data & ML:** Pandas, Scikit-learn (sklearn), XGBoost, LightGBM
* **Visualization:** Plotly
* **AI/LLM:** Groq, python-dotenv

## 📁 Project Structure
Auto-ML/
├── .env                  # Stores API keys (GROQ_API_KEY)
├── app.py                # The main Streamlit application
├── requirements.txt      # All Python dependencies
├── style.css             # Custom CSS for the dark-mode UI
└── src/
    ├── preprocessing/
    │   └── core.py       # Data cleaning, encoding, scaling functions
    ├── training/
    │   ├── train.py      # Model training & evaluation logic
    │   └── tune.py       # Hyperparameter tuning grids & functions
    ├── ui/
    │   ├── chatbot.py    # UI for the AI chatbot page
    │   ├── evaluate.py   # UI for the model evaluation page
    │   ├── explore.py    # UI for the data exploration page
    │   ├── insights.py   # UI for the AI insights page
    │   ├── preprocess.py # UI for the preprocessing page
    │   └── train.py      # UI for the model training page
    └── utils/
        └── insights.py   # Groq client configuration & API call logic


## 🚀 Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/Auto-ML.git](https://github.com/your-username/Auto-ML.git)
    cd Auto-ML
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your API keys:**
    * Get a free API key from [Groq](https://groq.com/).
    * Create a file named `.env` in the `Auto-ML` root folder.
    * Add your key to the file:
        ```
        GROQ_API_KEY="your_api_key_here"
        ```

## 🎮 How to Use

1.  **Run the app:**
    ```bash
    streamlit run app.py
    ```
2.  **Upload:** Go to the "Upload" tab and provide your CSV file.
3.  **Preprocess:** Go to the "Preprocess" tab, select your target variable, and click "Apply Preprocessing."
4.  **Train:** Go to the "Train" tab, select your models, and click "Start Training."
5.  **Evaluate:** Go to the "Evaluate" tab.
    * View the full results table.
    * Select a model to see detailed plots.
    * Click "Tune" to optimize the best model.
6.  **Insights:** Go to the "Insights" tab and click the buttons to get an AI analysis.
7.  **Chatbot:** Go to the "Chatbot" tab and ask questions about your results.
