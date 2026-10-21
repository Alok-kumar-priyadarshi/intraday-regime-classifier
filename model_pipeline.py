import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

# Create models directory
os.makedirs("models", exist_ok=True)

def fetch_data():
    print("Fetching Nifty 50 Hourly Data...")
    # Fetching Nifty 50 index data
    nifty = yf.download("^NSEI", period="730d", interval="1h")
    # Flatten multi-level columns if using newer yfinance versions
    if isinstance(nifty.columns, pd.MultiIndex):
        nifty.columns = nifty.columns.get_level_values(0)
    nifty.dropna(inplace=True)
    return nifty

def engineer_features_and_labels(df):
    print("Engineering features and labels...")
    df = df.copy()
    
    # 1. Feature Engineering
    df['Returns'] = df['Close'].pct_change()
    df['Rolling_Std_5'] = df['Returns'].rolling(window=5).std()
    df['Rolling_Std_15'] = df['Returns'].rolling(window=15).std()
    
    # Volume Shocks (if volume data is 0 for index, we use price proxy, but let's try standard volume)
    df['Volume_SMA'] = df['Volume'].rolling(window=15).mean()
    df['Volume_Shock'] = np.where(df['Volume_SMA'] == 0, 0, df['Volume'] / df['Volume_SMA'])
    
    # Cyclic time-of-day parameters
    df['Hour'] = df.index.hour
    df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    
    # 2. Ground Truth Labeling (Look forward 3 hours to define regime)
    df['Future_High'] = df['High'].shift(-3).rolling(window=3).max()
    df['Future_Low'] = df['Low'].shift(-3).rolling(window=3).min()
    df['Future_Range_Pct'] = (df['Future_High'] - df['Future_Low']) / df['Close']
    
    df.dropna(inplace=True)
    
    # Define thresholds for Quiet, Normal, Chaotic based on quantiles
    q30 = df['Future_Range_Pct'].quantile(0.30)
    q80 = df['Future_Range_Pct'].quantile(0.80)
    
    conditions = [
        (df['Future_Range_Pct'] <= q30),
        (df['Future_Range_Pct'] > q30) & (df['Future_Range_Pct'] <= q80),
        (df['Future_Range_Pct'] > q80)
    ]
    choices = ['Quiet', 'Normal', 'Chaotic']
    df['Regime'] = np.select(conditions, choices, default='Normal')
    
    return df

def train_model(df):
    print("Training ML Classifier...")
    features = ['Rolling_Std_5', 'Rolling_Std_15', 'Volume_Shock', 'Hour_Sin', 'Hour_Cos']
    X = df[features]
    y = df['Regime']
    
    # Split without shuffling to respect time series
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)
    
    print("\nModel Evaluation on Test Data:")
    print(classification_report(y_test, model.predict(X_test)))
    
    # Save the model
    joblib.dump(model, "models/regime_classifier.pkl")
    print("Model saved to models/regime_classifier.pkl")

if __name__ == "__main__":
    raw_data = fetch_data()
    processed_data = engineer_features_and_labels(raw_data)
    train_model(processed_data)
