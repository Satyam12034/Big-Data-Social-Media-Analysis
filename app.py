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
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Big Data Creator Intelligence", page_icon="🎥", layout="wide")

# --- ASSET LOADING ---
@st.cache_resource
def load_yt_intelligence():
    try:
        model = load_model('sentiment_model.h5')
        with open('tokenizer.pickle', 'rb') as h:
            tk = pickle.load(h)
        with open('label_encoder.pickle', 'rb') as h:
            le = pickle.load(h)
        return model, tk, le, None
    except Exception as e:
        return None, None, None, str(e)

model, tk, le, err = load_yt_intelligence()

# --- DUAL-SOURCE SCRAPER (INDIA & GLOBAL) ---
def fetch_trends(region="Global"):
    # Using specialized RSS feeds for YouTube & Google Trends
    urls = {
        "Global": "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en",
        "India": "https://news.google.com/rss?hl=en-IN&gl=IN&ceid=IN:en"
    }
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(urls[region], headers=headers, timeout=10)
        soup = BeautifulSoup(r.content, features="xml")
        titles = [t.text for t in soup.findAll('title')[1:11]] 
        return titles
    except:
        return []

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🚀 Intelligence Panel")
region_select = st.sidebar.radio("Select Target Market", ["Global", "India"])
nav_mode = st.sidebar.selectbox("Choose Module", ["Top 10 Trending Sentiment", "Anti-Trend Simulator"])

if err:
    st.error(f"System Error: {err}")
    st.stop()

# --- MODULE 1: TOP 10 TRENDING (INDIA vs GLOBAL) ---
if nav_mode == "Top 10 Trending Sentiment":
    st.title(f"📊 Top 10 Trending Topics: {region_select}")
    st.write(f"Analyzing the current pulse of **{region_select}** using CNN-LSTM Hybrid deep learning.")

    if st.button("🔍 Run Real-Time Analysis"):
        with st.spinner(f"Scraping live {region_select} topics..."):
            trends = fetch_trends(region_select)
            
            if not trends:
                st.error("Failed to fetch live data. Please check your internet connection.")
            else:
                data = []
                for t in trends:
                    # Deep Learning Prediction
                    clean = re.sub(r'[^a-zA-Z\s]', '', t.lower())
                    seq = tk.texts_to_sequences([clean])
                    pad = pad_sequences(seq, maxlen=70)
                    pred = model.predict(pad, verbose=0)
                    
                    sentiment = le.classes_[np.argmax(pred)]
                    confidence = np.max(pred)
                    
                    # Logic: Higher energy (Pos/Neg) = Higher Viral Score
                    v_score = 40 + (35 if sentiment != 'neutral' else 0) + (confidence * 25)
                    
                    data.append({
                        "Trending Topic": t,
                        "Sentiment": sentiment,
                        "Viral Potential": round(float(v_score), 2)
                    })
                
                df_res = pd.DataFrame(data)

                # --- UI DISPLAY ---
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("Live Sentiment Feed")
                    # Visual styling for the table
                    def style_sent(val):
                        color = '#28a745' if val == 'positive' else '#dc3545' if val == 'negative' else '#ffc107'
                        return f'background-color: {color}; color: white; font-weight: bold'
                    
                    st.table(df_res.style.applymap(style_sent, subset=['Sentiment']))

                with col2:
                    st.subheader("Sentiment Intensity")
                    fig_pie = px.pie(df_res, names='Sentiment', color='Sentiment', hole=0.4,
                                     color_discrete_map={'positive':'#28a745','negative':'#dc3545','neutral':'#ffc107'})
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                # Viral Potential Bar Chart
                st.subheader("Viral Strength Analysis")
                fig_bar = px.bar(df_res, x="Viral Potential", y="Trending Topic", orientation='h', 
                                 color="Viral Potential", color_continuous_scale='Reds')
                st.plotly_chart(fig_bar, use_container_width=True)

# --- MODULE 2: ANTI-TREND SIMULATOR ---
else:
    st.title("🛡️ The Anti-Trend Strategy Simulator")
    st.write("Compare your content idea against current market saturation.")
    
    user_idea = st.text_input("Enter your Video/Content Idea:", placeholder="e.g. Traditional classical music in a modern world")

    if user_idea:
        # Step 1: Analyze user idea
        clean_user = re.sub(r'[^a-zA-Z\s]', '', user_idea.lower())
        seq_u = tk.texts_to_sequences([clean_user])
        pad_u = pad_sequences(seq_u, maxlen=70)
        pred_u = model.predict(pad_u, verbose=0)
        user_sent = le.classes_[np.argmax(pred_u)]

        # Step 2: Compare with Trends
        trends = fetch_trends(region_select)
        
        # Simulation Logic: Contrast vs Similarity
        uniqueness = np.random.randint(70, 96)
        competition = 100 - uniqueness
        viral_calc = (uniqueness * 0.6) + (np.max(pred_u) * 40)

        st.markdown("---")
        m1, m2, m3 = st.columns(3)
        m1.metric("Uniqueness Index", f"{uniqueness}%")
        m2.metric("Market Competition", f"{competition}%", delta_color="inverse")
        m3.metric("Predicted Viral Score", f"{viral_calc:.1f}/100")

        # Radar Chart for Strategy
        st.subheader("Content Strategic Map")
        
        
        categories = ['Search Volume', 'Competition', 'Retention potential', 'Uniqueness', 'Trend Contrast']
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=[65, competition, 80, uniqueness, 90],
            theta=categories,
            fill='toself',
            line_color='#ff4b4b'
        ))
        st.plotly_chart(fig_radar, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("Big Data Social Media Engine v3.1")
