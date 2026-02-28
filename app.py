import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import plotly.express as px

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Big Data Social Media Analysis",
    page_icon="📊",
    layout="wide"
)

# --- CUSTOM CSS FOR PROFESSIONAL LOOK ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0px 2px 10px rgba(0,0,0,0.05); }
    .prediction-box { padding: 20px; border-radius: 10px; text-align: center; color: white; font-weight: bold; font-size: 24px; }
    </style>
    """, unsafe_allow_html=True)

# --- ASSET LOADING WITH ERROR HANDLING ---
@st.cache_resource
def load_assets():
    assets = {}
    try:
        # Load the Model
        assets['model'] = load_model('sentiment_model.h5')
        
        # Load Tokenizer
        with open('tokenizer.pickle', 'rb') as handle:
            assets['tokenizer'] = pickle.load(handle)
            
        # Load Label Encoder
        with open('label_encoder.pickle', 'rb') as handle:
            assets['le'] = pickle.load(handle)
            
        # Robust Data Loading (Handling Encoding & Path issues)
        file_path = 'test.csv'
        if os.path.exists(file_path):
            try:
                assets['df'] = pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                assets['df'] = pd.read_csv(file_path, encoding='ISO-8859-1')
        else:
            assets['df'] = None
            
        return assets, None
    except Exception as e:
        return None, str(e)

# Initialize assets
data_assets, error_msg = load_assets()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🛠️ Project Controls")
st.sidebar.markdown("---")
app_mode = st.sidebar.selectbox("Choose a Page", ["Home & Predictor", "Big Data Insights"])

if error_msg:
    st.error(f"⚠️ **Initialization Error:** {error_msg}")
    st.info("Ensure `sentiment_model.h5`, `tokenizer.pickle`, and `test.csv` are in the root directory.")
    st.stop()

# --- PAGE 1: PREDICTOR ---
if app_mode == "Home & Predictor":
    st.title("📊 Big Data Analysis of Social Media Trends")
    st.write("This application uses a Deep Learning model trained from scratch to analyze sentiment and visualize global social media behavior.")
    
    st.markdown("### 🔍 Real-time Sentiment Predictor")
    user_input = st.text_area("What's on your mind? (Enter a tweet or post)", 
                              placeholder="Type something like 'I love the new updates!'", 
                              height=100)

    if st.button("Analyze Sentiment"):
        if user_input:
            # Pre-processing (must match training)
            clean_text = user_input.lower()
            clean_text = re.sub(r'[^\w\s]', '', clean_text)
            
            # Prediction Logic
            seq = data_assets['tokenizer'].texts_to_sequences([clean_text])
            padded = pad_sequences(seq, maxlen=50) # Matching Training MAX_SEQUENCE_LENGTH
            
            prediction = data_assets['model'].predict(padded)
            class_idx = np.argmax(prediction)
            label = data_assets['le'].classes_[class_idx]
            confidence = np.max(prediction) * 100

            # Professional Result Display
            col1, col2 = st.columns([2, 1])
            with col1:
                bg_color = "#28a745" if label == 'positive' else "#dc3545" if label == 'negative' else "#ffc107"
                st.markdown(f"""<div class='prediction-box' style='background-color: {bg_color};'>
                            Detected Sentiment: {label.upper()}</div>""", unsafe_allow_html=True)
            
            with col2:
                st.metric("Model Confidence", f"{confidence:.1f}%")
        else:
            st.warning("Please enter some text first!")

# --- PAGE 2: BIG DATA INSIGHTS ---
else:
    st.title("📈 Global Trend Dashboard")
    
    if data_assets['df'] is not None:
        df = data_assets['df']
        
        # Row 1: Key Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Records Analyzed", len(df))
        m2.metric("Unique Countries", df['Country'].nunique())
        m3.metric("Top Sentiment", df['sentiment'].mode()[0].capitalize())
        
        st.markdown("---")
        
        # Row 2: Charts
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("Sentiment Distribution")
            fig1 = px.pie(df, names='sentiment', hole=0.4, color='sentiment',
                          color_discrete_map={'positive':'#28a745', 'negative':'#dc3545', 'neutral':'#ffc107'})
            st.plotly_chart(fig1, use_container_width=True)
            
        with c2:
            st.subheader("Activity by Time of Day")
            fig2 = px.histogram(df, x="Time of Tweet", color="sentiment", barmode="group")
            st.plotly_chart(fig2, use_container_width=True)
            
        # Row 3: Demographics
        st.subheader("Sentiment Trends by Age Group")
        fig3 = px.bar(df, x="Age of User", y="Population -2020", color="sentiment", 
                      title="User Reach vs Demographic Sentiment")
        st.plotly_chart(fig3, use_container_width=True)
        
    else:
        st.error("Data file 'test.csv' not found. Dashboard cannot be generated.")

# --- FOOTER ---
st.markdown("---")
st.caption("College Project | Big Data Sentiment Engine | Build version 2.0.1")
