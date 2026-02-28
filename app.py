import streamlit as st
import pandas as pd
import numpy as np
import pickle, requests, re, os
from bs4 import BeautifulSoup
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Sentira: Multi-Domain AI", layout="wide", page_icon="📈")

# --- PROFESSIONAL STYLING ---
st.markdown("""
    <style>
    .stTable { background-color: white; border-radius: 10px; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }
    h1 { color: #1E3A8A; font-family: 'Segoe UI'; }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD ENGINE ---
@st.cache_resource
def load_engine():
    try:
        m = load_model('sentiment_model.h5')
        with open('tokenizer.pickle', 'rb') as h: tk = pickle.load(h)
        with open('label_encoder.pickle', 'rb') as h: le = pickle.load(h)
        return m, tk, le, None
    except Exception as e:
        return None, None, None, str(e)

model, tk, le, load_err = load_engine()

# --- HELPER FUNCTIONS ---
def scrape_data(url):
    try:
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        soup = BeautifulSoup(r.content, features="xml")
        return [item.title.text for item in soup.findAll('item')[:10]]
    except: return []

def analyze_titles(titles):
    results = []
    for t in titles:
        clean = re.sub(r'[^a-zA-Z\s]', '', t.lower())
        seq = tk.texts_to_sequences([clean])
        pad = pad_sequences(seq, maxlen=100)
        pred = model.predict(pad, verbose=0)
        sent = le.classes_[np.argmax(pred)]
        results.append({"Headline/Post": t, "Sentiment": sent, "Intensity": np.max(pred)})
    return pd.DataFrame(results)

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("📊 Intelligence Domains")
category = st.sidebar.selectbox("Choose Domain", 
    ["Global & India Pulse", "Sports (Cricket/Chess)", "Bollywood Buzz", "Mental Health Analysis"])

if load_err:
    st.error(f"Asset Load Error: {load_err}")
    st.stop()

# --- MODULE 1: GLOBAL/INDIA ---
if category == "Global & India Pulse":
    st.title("🌐 Real-Time Global & National Pulse")
    region = st.radio("Select Target Region", ["Global", "India"], horizontal=True)
    url = "https://news.google.com/rss?hl=en-IN&gl=IN&ceid=IN:en" if region == "India" else "https://news.google.com/rss"
    
    if st.button("Generate Pulse Report"):
        df = analyze_titles(scrape_data(url))
        st.subheader("Live Headlines & Predicted Sentiment")
        st.table(df.style.applymap(lambda x: 'color: #28a745' if x=='positive' else 'color: #dc3545' if x=='negative' else 'color: #ffc107', subset=['Sentiment']))
        
        # Graphs Below
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.pie(df, names='Sentiment', hole=0.5, title="Aggregate Mood Share", 
                                 color='Sentiment', color_discrete_map={'positive':'#28a745','negative':'#dc3545','neutral':'#ffc107'}), use_container_width=True)
        with c2:
            st.plotly_chart(px.bar(df, x="Sentiment", y="Intensity", color="Sentiment", title="Sentiment Confidence Levels"), use_container_width=True)

# --- MODULE 2: SPORTS ---
elif category == "Sports (Cricket/Chess)":
    st.title("🏏 Sports Intelligence: Performance & Buzz")
    url = "https://news.google.com/rss/search?q=Cricket+Chess+IPL+Grandmaster"
    
    if st.button("Fetch Sports Sentiment"):
        df = analyze_titles(scrape_data(url))
        st.table(df)
        
        st.markdown("---")
        st.subheader("Sports Energy Heatmap")
        fig = px.density_heatmap(df, x="Sentiment", y="Intensity", title="Energy Intensity of Current Matches", color_continuous_scale="Viridis")
        st.plotly_chart(fig, use_container_width=True)

# --- MODULE 3: BOLLYWOOD ---
elif category == "Bollywood Buzz":
    st.title("🎬 Bollywood Release & Review Tracker")
    url = "https://news.google.com/rss/search?q=Bollywood+Movie+Review+Box+Office+Buzz"
    
    if st.button("Analyze Movie Buzz"):
        df = analyze_titles(scrape_data(url))
        st.table(df)
        
        st.markdown("---")
        st.subheader("Pre-Release vs Post-Release Sentiment")
        fig = px.area(df, x="Headline/Post", y="Intensity", color="Sentiment", title="Buzz Velocity Mapping")
        st.plotly_chart(fig, use_container_width=True)

# --- MODULE 4: MENTAL HEALTH ---
elif category == "Mental Health Analysis":
    st.title("🧠 Mental Health & Wellness Discourse")
    st.info("Tracking societal shifts in Mental Health conversations post-COVID.")
    url = "https://news.google.com/rss/search?q=Mental+Health+Post+Covid+Social+Media+Reddit"
    
    if st.button("Analyze Conversations"):
        df = analyze_titles(scrape_data(url))
        st.table(df)
        
        st.markdown("---")
        pos_rate = (len(df[df['Sentiment'] == 'positive']) / len(df)) * 100
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = pos_rate,
            title = {'text': "Community Wellness Index (%)"},
            gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "#1E3A8A"}, 
                     'steps' : [{'range': [0, 40], 'color': "#FFCCCC"}, {'range': [40, 70], 'color': "#FFFFCC"}, {'range': [70, 100], 'color': "#CCFFCC"}]}
        ))
        st.plotly_chart(fig, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("Sentira AI Engine v4.0 | Project Submission")
