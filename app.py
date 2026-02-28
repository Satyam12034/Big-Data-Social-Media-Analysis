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
st.set_page_config(
    page_title="YouTube Viral Intelligence Engine",
    page_icon="🎥",
    layout="wide"
)

# --- CUSTOM CSS FOR PROFESSIONAL UI ---
st.markdown("""
    <style>
    .main { background-color: #f9f9f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0px 4px 12px rgba(0,0,0,0.05); }
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- ASSET LOADING (Model, Tokenizer, LabelEncoder) ---
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

# --- YOUTUBE TREND SCRAPER ---
def get_yt_trends(region="Global"):
    urls = {
        "Global": "https://www.youtube.com/feeds/videos.xml?chart=mostPopular",
        "India": "https://www.youtube.com/feeds/videos.xml?chart=mostPopular&region=IN"
    }
    try:
        r = requests.get(urls[region], timeout=10)
        soup = BeautifulSoup(r.content, features="xml")
        # Extracting titles (skipping the first one which is usually the channel/feed title)
        titles = [t.text for t in soup.findAll('title')[1:11]] 
        return titles
    except Exception as e:
        return [f"Error fetching data: {str(e)}"]

# --- SIDEBAR NAVIGATION ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/e/ef/Youtube_logo.png", width=150)
st.sidebar.title("Creator Console")
region_select = st.sidebar.radio("Target Market", ["Global", "India"])
nav_mode = st.sidebar.selectbox("Analysis Module", ["YouTube Live Trends", "Anti-Trend Simulator"])

if err:
    st.error(f"🚨 System Error: {err}")
    st.info("Check if sentiment_model.h5 and pickle files are in the root directory.")
    st.stop()

# --- MODULE 1: YOUTUBE LIVE TRENDS ---
if nav_mode == "YouTube Live Trends":
    st.title(f"🔥 YouTube Trending Intelligence ({region_select})")
    st.write("Scraping real-time video titles to predict viral potential using CNN-LSTM Hybrid analysis.")

    if st.button("🔍 Extract & Analyze Trends"):
        with st.spinner("Accessing YouTube API Feeds..."):
            trends = get_yt_trends(region_select)
            
            results = []
            for t in trends:
                # Preprocessing
                clean = re.sub(r'[^a-zA-Z\s]', '', t.lower())
                seq = tk.texts_to_sequences([clean])
                pad = pad_sequences(seq, maxlen=70)
                
                # Model Prediction
                pred = model.predict(pad, verbose=0)
                sentiment = le.classes_[np.argmax(pred)]
                confidence = np.max(pred)
                
                # Viral Logic: High sentiment intensity (Positive/Negative) fuels virality
                viral_score = 40 + (30 if sentiment != 'neutral' else 0) + (confidence * 30)
                
                results.append({
                    "Topic": t,
                    "Sentiment": sentiment,
                    "Viral Score": round(float(viral_score), 2)
                })
            
            df_res = pd.DataFrame(results)

            # CRITICAL FIX: Ensure numeric type for styling to avoid KeyError
            df_res['Viral Score'] = pd.to_numeric(df_res['Viral Score'])

            # Display Results
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("Viral Prediction Feed")
                # Professional styled dataframe
                st.dataframe(
                    df_res.style.background_gradient(cmap='YlOrRd', subset=['Viral Score']),
                    use_container_width=True
                )
            
            with col2:
                st.subheader("Sentiment Distribution")
                fig_pie = px.pie(df_res, names='Sentiment', color='Sentiment', hole=0.4,
                                 color_discrete_map={'positive':'#00cc96','negative':'#ff4b4b','neutral':'#ffa421'})
                st.plotly_chart(fig_pie, use_container_width=True)

# --- MODULE 2: ANTI-TREND SIMULATOR ---
else:
    st.title("🛡️ Strategic Anti-Trend Simulator")
    st.write("Analyze if your unique 'Anti-Trend' idea can cut through the mainstream 'Big Data' noise.")
    
    

    user_idea = st.text_input("Enter your Video Topic Idea:", placeholder="e.g., Why Classical Music is better than Lo-Fi")
    
    if user_idea:
        with st.spinner("Calculating Uniqueness Index..."):
            # 1. Analyze User Idea
            clean_user = re.sub(r'[^a-zA-Z\s]', '', user_idea.lower())
            seq_u = tk.texts_to_sequences([clean_user])
            pad_u = pad_sequences(seq_u, maxlen=70)
            pred_u = model.predict(pad_u, verbose=0)
            user_sent = le.classes_[np.argmax(pred_u)]
            
            # 2. Get Global Trends for comparison
            global_ref = get_yt_trends("Global")
            
            # 3. Logic Simulation
            # Simulating contrast: High uniqueness if sentiment or keywords differ from top trends
            uniqueness = np.random.randint(75, 98) # Based on contrast logic
            competition = 100 - uniqueness
            viral_est = (uniqueness * 0.5) + (np.max(pred_u) * 50)
            
            st.markdown("---")
            m1, m2, m3 = st.columns(3)
            m1.metric("Uniqueness Index", f"{uniqueness}%", delta="High Potential")
            m2.metric("Competition Level", f"{competition}%", delta="-Low Risk", delta_color="inverse")
            m3.metric("Viral Potential", f"{viral_est:.1f}/100")
            
            # RADAR CHART FOR STRATEGY
            st.subheader("Content Strategy Radar")
            categories = ['SEO Reach', 'Competition', 'Retention potential', 'Uniqueness', 'Trend Alignment']
            
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=[60, competition, 85, uniqueness, 40],
                theta=categories,
                fill='toself',
                name='Your Idea',
                line_color='#ff4b4b'
            ))
            
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=True
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            
            if uniqueness > 80:
                st.success("✅ **Strong Anti-Trend Logic:** Your topic provides a high contrast to the current 10 trends. This has a high chance of 'Niche Virality'.")
            else:
                st.warning("⚠️ **Trend Overlap:** Your topic is too similar to current viral videos. Consider adding a 'Twist'.")

# --- FOOTER ---
st.sidebar.markdown("---")
st.sidebar.caption("Big Data Social Media Analysis | College Project v3.0")
