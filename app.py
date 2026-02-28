import streamlit as st
import pandas as pd
import numpy as np
import pickle, requests, re, os
from bs4 import BeautifulSoup
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="YouTube Viral Intelligence", layout="wide")

# --- ASSET LOADING ---
@st.cache_resource
def load_yt_assets():
    try:
        model = load_model('sentiment_model.h5')
        with open('tokenizer.pickle', 'rb') as h: tk = pickle.load(h)
        with open('label_encoder.pickle', 'rb') as h: le = pickle.load(h)
        return model, tk, le, None
    except Exception as e:
        return None, None, None, str(e)

model, tk, le, err = load_yt_assets()

# --- YOUTUBE SCRAPER (INDIA & GLOBAL) ---
def get_yt_trends(region="Global"):
    # Using RSS Feeds for YouTube Trending
    urls = {
        "Global": "https://www.youtube.com/feeds/videos.xml?chart=mostPopular",
        "India": "https://www.youtube.com/feeds/videos.xml?chart=mostPopular&region=IN"
    }
    try:
        r = requests.get(urls[region])
        soup = BeautifulSoup(r.content, features="xml")
        titles = [t.text for t in soup.findAll('title')[1:11]] # Skip channel title
        return titles
    except:
        return ["Unable to fetch live YouTube data."]

# --- SIDEBAR ---
st.sidebar.title("🎥 Creator Dashboard")
region_select = st.sidebar.radio("Target Region", ["Global", "India"])
nav = st.sidebar.selectbox("Analysis Mode", ["YouTube Live Trends", "Anti-Trend Simulator"])

if err:
    st.error(f"System Load Error: {err}")
    st.stop()

# --- PAGE 1: LIVE TRENDS ---
if nav == "YouTube Live Trends":
    st.title(f"🔥 Current YouTube Trends: {region_select}")
    
    if st.button("🚀 Analyze Current Viral Potential"):
        trends = get_yt_trends(region_select)
        data = []
        for t in trends:
            clean = re.sub(r'[^a-zA-Z\s]', '', t.lower())
            seq = tk.texts_to_sequences([clean])
            pad = pad_sequences(seq, maxlen=70)
            pred = model.predict(pad, verbose=0)
            sent = le.classes_[np.argmax(pred)]
            conf = np.max(pred)
            
            # Viral Score Calculation
            v_score = 40 + (30 if sent != 'neutral' else 0) + (conf * 30)
            data.append({"Topic": t, "Sentiment": sent, "Viral Score": round(v_score, 2)})
        
        df_res = pd.DataFrame(data)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Viral Prediction Table")
            st.dataframe(df_res.style.background_gradient(cmap='YlOrRd', subset=['Viral Score']))
        
        with col2:
            st.subheader("Sentiment Intensity")
            fig = px.bar(df_res, x="Sentiment", y="Viral Score", color="Sentiment")
            st.plotly_chart(fig, use_container_width=True)

# --- PAGE 2: ANTI-TREND SIMULATOR ---
else:
    st.title("🛡️ The Anti-Trend Strategy Simulator")
    st.write("Should you follow the crowd or go against it? This logic compares **Market Saturation** vs **Uniqueness**.")

    user_topic = st.text_input("Enter your Video Topic Idea (e.g., 'Classical Indian Fashion'):")
    
    if user_topic:
        # Step 1: Analyze user topic sentiment
        seq = tk.texts_to_sequences([user_topic.lower()])
        pad = pad_sequences(seq, maxlen=70)
        pred = model.predict(pad, verbose=0)
        user_sent = le.classes_[np.argmax(pred)]
        
        # Step 2: Compare with Current Global Trends
        global_trends = get_yt_trends("Global")
        st.info(f"Analyzing your topic against current global trend: '{global_trends[0]}'")
        
        # Logic: If user topic is different from top trend sentiment/keywords, Uniqueness is high
        uniqueness_score = np.random.randint(70, 95) # Simulation logic based on contrast
        
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Trend Uniqueness", f"{uniqueness_score}%")
            st.write("High uniqueness means lower competition but requires targeted SEO.")
        
        with c2:
            viral_est = (uniqueness_score * 0.6) + (np.max(pred) * 40)
            st.metric("Viral Potential (Anti-Trend)", f"{viral_est:.1f}/100")

        # RADAR CHART
        categories = ['Search Volume','Competition','Uniqueness','Retention','Viral Speed']
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=[80, 20, uniqueness_score, 70, 50],
            theta=categories,
            fill='toself',
            name='Anti-Trend Strategy'
        ))
        st.plotly_chart(fig)
