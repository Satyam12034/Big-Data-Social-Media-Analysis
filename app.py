import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
import re
import os
from bs4 import BeautifulSoup
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import plotly.express as px

# --- APP CONFIGURATION ---
st.set_page_config(page_title="Big Data Intelligence", page_icon="🌐", layout="wide")

# --- ASSET LOADING WITH ROBUST ERROR HANDLING ---
@st.cache_resource
def load_all_assets():
    try:
        model = load_model('sentiment_model.h5')
        with open('tokenizer.pickle', 'rb') as h:
            tk = pickle.load(h)
        with open('label_encoder.pickle', 'rb') as h:
            le = pickle.load(h)
        
        # Robust CSV Load (Encoding & Path Fix)
        if os.path.exists('test.csv'):
            try:
                raw_df = pd.read_csv('test.csv', encoding='utf-8')
            except:
                raw_df = pd.read_csv('test.csv', encoding='ISO-8859-1')
        else:
            raw_df = pd.DataFrame()
            
        return model, tk, le, raw_df, None
    except Exception as e:
        return None, None, None, None, str(e)

model, tk, le, raw_df, load_error = load_all_assets()

# --- WEB SCRAPER ---
def get_current_trends():
    try:
        url = "https://news.google.com/rss"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, features="xml")
        return [item.title.text for item in soup.findAll('item')[:10]]
    except:
        return []

# --- SIDEBAR NAV ---
st.sidebar.title("📊 Trend Controller")
mode = st.sidebar.selectbox("Select Dashboard", ["Live Trend Extraction", "Big Data Analytics"])

if load_error:
    st.error(f"Critical System Error: {load_error}")
    st.info("Check if model files and test.csv are in the root directory.")
    st.stop()

# --- PAGE 1: LIVE SCRAPER & ANALYSIS ---
if mode == "Live Trend Extraction":
    st.title("🌐 Live Social Media & News Pulse")
    st.write("This module scrapes live global data and processes it through the CNN-LSTM Hybrid Neural Network.")

    if st.button("🔍 Extract & Analyze Live Trends"):
        headlines = get_current_trends()
        if not headlines:
            st.error("Could not reach news servers.")
        else:
            results = []
            for text in headlines:
                # Clean and Predict
                clean_text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
                seq = tk.texts_to_sequences([clean_text])
                pad = pad_sequences(seq, maxlen=65)
                pred = model.predict(pad, verbose=0)
                
                label = le.classes_[np.argmax(pred)]
                results.append({"Trending Topic": text, "Sentiment": label, "Confidence": np.max(pred)})
            
            res_df = pd.DataFrame(results)

            # Display UI
            c1, c2 = st.columns([2, 1])
            with c1:
                st.subheader("Live Feed Results")
                def style_sentiment(v):
                    c = '#28a745' if v == 'positive' else '#dc3545' if v == 'negative' else '#ffc107'
                    return f'background-color: {c}; color: white; font-weight: bold'
                st.table(res_df.style.applymap(style_sentiment, subset=['Sentiment']))
            
            with c2:
                st.subheader("Global Mood Overview")
                fig = px.pie(res_df, names='Sentiment', color='Sentiment', hole=0.4,
                             color_discrete_map={'positive':'#28a745','negative':'#dc3545','neutral':'#ffc107'})
                st.plotly_chart(fig, use_container_width=True)

# --- PAGE 2: HISTORICAL ANALYTICS ---
else:
    st.title("📂 Historical Big Data Insights")
    if raw_df.empty:
        st.warning("No historical data found (test.csv missing).")
    else:
        st.write("Analyzing demographic relationships across the existing Big Data corpus.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Sentiment vs Age Group")
            fig1 = px.histogram(raw_df, x="Age of User", color="sentiment", barmode="group")
            st.plotly_chart(fig1)
        
        with col2:
            st.subheader("Geographic Sentiment Map")
            fig2 = px.choropleth(raw_df, locations="Country", locationmode='country names', color="sentiment")
            st.plotly_chart(fig2)

st.markdown("---")
st.caption("Advanced Big Data Project | CNN-LSTM Hybrid Implementation | [Your Name]")
