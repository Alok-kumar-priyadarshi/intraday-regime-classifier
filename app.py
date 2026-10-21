import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from twilio.rest import Client
import datetime

# --- CONFIGURATION ---
st.set_page_config(page_title="Nifty 50 Risk Radar", layout="wide")

# TWILIO SETUP (Replace with your free Twilio Trial credentials for the demo)
TWILIO_SID = 'YOUR_TWILIO_SID' 
TWILIO_TOKEN = 'YOUR_TWILIO_AUTH_TOKEN'
TWILIO_FROM = 'YOUR_TWILIO_PHONE_NUMBER'
USER_PHONE = 'YOUR_VERIFIED_PHONE_NUMBER'

@st.cache_resource
def load_model():
    try:
        return joblib.load("models/regime_classifier.pkl")
    except:
        st.error("Model not found. Please run model_pipeline.py first.")
        return None

def fetch_latest_data():
    nifty = yf.download("^NSEI", period="7d", interval="1h")
    if isinstance(nifty.columns, pd.MultiIndex):
        nifty.columns = nifty.columns.get_level_values(0)
    return nifty.dropna()

def prepare_live_features(df):
    df['Returns'] = df['Close'].pct_change()
    df['Rolling_Std_5'] = df['Returns'].rolling(window=5).std()
    df['Rolling_Std_15'] = df['Returns'].rolling(window=15).std()
    df['Volume_SMA'] = df['Volume'].rolling(window=15).mean()
    df['Volume_Shock'] = np.where(df['Volume_SMA'] == 0, 0, df['Volume'] / df['Volume_SMA'])
    df['Hour'] = df.index.hour
    df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    return df.dropna().tail(1)

def send_sms_alert(regime):
    try:
        client = Client(TWILIO_SID, TWILIO_TOKEN)
        message = client.messages.create(
            body=f"🚨 ALERT: Nifty 50 is entering a {regime} regime. High volatility expected. Protect your capital.",
            from_=TWILIO_FROM,
            to=USER_PHONE
        )
        return True
    except Exception as e:
        return False

# --- UI DASHBOARD ---
st.title("📈 Real-Time Intraday Volatility Regime Classifier")
st.markdown("An ML-powered Risk Radar to warn retail traders of impending market shocks.")

model = load_model()
if model:
    df = fetch_latest_data()
    latest_features = prepare_live_features(df)
    
    if not latest_features.empty:
        # Predict Regime
        X_live = latest_features[['Rolling_Std_5', 'Rolling_Std_15', 'Volume_Shock', 'Hour_Sin', 'Hour_Cos']]
        prediction = model.predict(X_live)[0]
        probabilities = model.predict_proba(X_live)[0]
        
        # UI Colors
        color_map = {"Quiet": "green", "Normal": "orange", "Chaotic": "red"}
        ui_color = color_map.get(prediction, "grey")
        
        # Display Risk Radar
        st.subheader("Current Market Status")
        st.markdown(
            f"<div style='background-color:{ui_color}; padding:20px; border-radius:10px; text-align:center; color:white;'>"
            f"<h1>{prediction.upper()} REGIME</h1>"
            f"</div>", 
            unsafe_allow_html=True
        )
        
        # SMS Trigger
        if prediction == "Chaotic":
            if st.button("Simulate Twilio SMS Alert"):
                success = send_sms_alert(prediction)
                if success:
                    st.success("SMS Alert Sent Successfully via Twilio!")
                else:
                    st.warning("Twilio credentials not configured. SMS failed.")
                    
        # Charts
        st.markdown("---")
        st.subheader("Nifty 50 Recent Trend")
        
        fig = go.Figure(data=[go.Candlestick(x=df.index,
                        open=df['Open'],
                        high=df['High'],
                        low=df['Low'],
                        close=df['Close'])])
        fig.update_layout(xaxis_rangeslider_visible=False, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Model Confidence")
        st.bar_chart(pd.DataFrame([probabilities], columns=model.classes_))
