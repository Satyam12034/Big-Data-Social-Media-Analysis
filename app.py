import streamlit as st
import pandas as pd
import numpy as np
import pickle, requests, re, os
from bs4 import BeautifulSoup
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import plotly.express as px
import plotly.graph_objects as go

# --- APP CONFIG ---
st.set_page_config(page_title="Sentira AI: Multi-Sport & Trend Intelligence", layout="wide")

@st.cache_resource
def load_engine():
    m = load_model('sentiment_model.h5')
    with open('tokenizer.pickle', 'rb') as h: tk = pickle.load(h)
    with open('label_encoder.pickle', 'rb') as h: le = pickle.load(h)
    return m, tk, le

model, tk, le = load_engine()

# --- UTILS (Yeh Function har section use karega) ---
def color_sent(v):
    c = '#28a745' if v == 'positive' else '#dc3545' if v == 'negative' else '#ffc107'
    return f'background-color: {c}; color: white; font-weight: bold'

# --- SCRAPER ---
def scrape_rss(url):
    try:
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        soup = BeautifulSoup(r.content, features="xml")
        return [item.title.text for item in soup.findAll('item')[:10]]
    except: return []

# --- ANALYSIS ENGINE ---
def run_analysis(titles):
    results = []
    for t in titles:
        clean = re.sub(r'[^a-zA-Z\s]', '', t.lower())
        seq = tk.texts_to_sequences([clean])
        pad = pad_sequences(seq, maxlen=100)
        pred = model.predict(pad, verbose=0)
        sent = le.classes_[np.argmax(pred)]
        results.append({"Headline": t, "Sentiment": sent, "Confidence": np.max(pred)})
    return pd.DataFrame(results)

# --- SIDEBAR ---
st.sidebar.title("🏆 Intelligence Hub")
domain = st.sidebar.selectbox("Select Domain", 
    ["All Sports News (Default)", "Global & India Pulse", "Bollywood Buzz", "Mental Health Analysis"])

# --- DOMAIN LOGIC ---
if domain == "All Sports News":
    st.title("🏆 World Sports Intelligence")
    st.write("Live tracking of Football, Cricket, Tennis, F1, and more.")
    
    # RSS for Global Sports
    sport_url = "https://news.google.com/rss/headlines/section/topic/SPORTS"
    
    if st.button("Fetch Live Sports Trends"):
        df = run_analysis(scrape_rss(sport_url))
        
        # 1. TABLE ON TOP
        st.subheader("Live Sports Feed")
        st.table(df.style.applymap(color_sent, subset=['Sentiment']))

        # 2. GRAPHS BELOW
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(px.pie(df, names='Sentiment', title="Overall Sports Mood", hole=0.4), use_container_width=True)
        with col2:
            st.plotly_chart(px.bar(df, x="Sentiment", y="Confidence", color="Sentiment", title="Analysis Confidence"), use_container_width=True)

elif domain == "Global & India Pulse":
    st.title("🌐 Global vs. India News")
    region = st.radio("Choose Region", ["India", "Global"], horizontal=True)
    url = "https://news.google.com/rss?hl=en-IN&gl=IN&ceid=IN:en" if region == "India" else "https://news.google.com/rss"
    
    if st.button("Run Analysis"):
        df = run_analysis(scrape_rss(url))
        # Yahan change kiya: color styling add kar di
        st.table(df.style.applymap(color_sent, subset=['Sentiment']))
        st.plotly_chart(px.sunburst(df, path=['Sentiment', 'Headline'], values='Confidence', title="Headline Impact Mapping"))

elif domain == "Bollywood Buzz":
    st.title("🎬 Bollywood Box Office & Social Buzz")
    url = "https://news.google.com/rss/search?q=Bollywood+Movie+Review+Box+Office"
    
    if st.button("Analyze Movie Buzz"):
        df = run_analysis(scrape_rss(url))
        # Yahan change kiya: color styling add kar di
        st.table(df.style.applymap(color_sent, subset=['Sentiment']))
        st.plotly_chart(px.area(df, x="Headline", y="Confidence", color="Sentiment", title="Buzz Velocity"))

elif domain == "Mental Health Analysis":
    st.title("🧠 Mental Health Conversations")
    url = "https://news.google.com/rss/search?q=Mental+Health+Awareness+Reddit+Twitter"
    
if st.button("Monitor Wellness Trends"):
        df = run_analysis(scrape_rss(url))
        st.table(df.style.applymap(color_sent, subset=['Sentiment']))
        
        # IMPROVED MATH: (Positive + 0.5 * Neutral) / Total
        pos_count = len(df[df['Sentiment']=='positive'])
        neu_count = len(df[df['Sentiment']=='neutral'])
        total = len(df)
        
        wellness_index = ((pos_count + (0.5 * neu_count)) / total) * 100
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = wellness_index,
            title = {'text': "Community Wellness Index (%)"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 40], 'color': "#ffcccb"}, # Red zone
                    {'range': [40, 70], 'color': "#fff9c4"}, # Yellow zone
                    {'range': [70, 100], 'color': "#c8e6c9"} # Green zone
                ]
            }
        ))
        st.plotly_chart(fig)

st.sidebar.markdown("---")
st.sidebar.caption("Big Data Social Engine v4.1")
