import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer

def handle_missing_values(df, numeric_strategy='mean', categorical_strategy='most_frequent'):
    """Handles missing values for numeric and categorical columns."""
    # This function is fine to run on the target too, so no changes needed here.
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    if not numeric_cols.empty:
        numeric_imputer = SimpleImputer(strategy=numeric_strategy)
        df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])
        
    if not categorical_cols.empty:
        categorical_imputer = SimpleImputer(strategy=categorical_strategy)
        df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])
        
    return df

def encode_categorical_features(df, target_variable, strategy='one_hot'):
    """Encodes categorical features, ignoring the target variable."""
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    # --- THIS IS THE FIX ---
    # Exclude the target variable from encoding
    if target_variable in categorical_cols:
        categorical_cols = categorical_cols.drop(target_variable)
    # --- END OF FIX ---

    if not categorical_cols.empty:
        if strategy == 'one_hot':
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
            
        elif strategy == 'label':
            le = LabelEncoder()
            for col in categorical_cols:
                df[col] = le.fit_transform(df[col])
            
    return df

def scale_numeric_features(df, target_variable, strategy='standard'):
    """Scales numeric features, ignoring the target variable."""
    
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    # --- THIS IS THE FIX ---
    # Exclude the target variable from scaling
    if target_variable in numeric_cols:
        numeric_cols = numeric_cols.drop(target_variable)
    # --- END OF FIX ---

    if not numeric_cols.empty:
        if strategy == 'standard':
            scaler = StandardScaler()
        elif strategy == 'min_max':
            scaler = MinMaxScaler()
        else:
            return df # No scaling
            
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        
    return df