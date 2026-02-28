import streamlit as st
import pandas as pd
import numpy as np
import pickle, requests, re, os
from bs4 import BeautifulSoup
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import plotly.express as px
import plotly.graph_objects as go

# --- CONFIG ---
st.set_page_config(page_title="Live India vs Global Trend Intel", layout="wide")

@st.cache_resource
def load_core_ai():
    m = load_model('sentiment_model.h5')
    with open('tokenizer.pickle', 'rb') as h: tk = pickle.load(h)
    with open('label_encoder.pickle', 'rb') as h: le = pickle.load(h)
    return m, tk, le

model, tk, le = load_core_ai()

# --- DYNAMIC SCRAPER ENGINE ---
def fetch_trends(region="Global"):
    # Different RSS feeds for Global vs India
    urls = {
        "Global": "https://news.google.com/rss",
        "India": "https://news.google.com/rss/headlines/section/topic/NATION?hl=en-IN&gl=IN&ceid=IN:en"
    }
    try:
        r = requests.get(urls[region])
        soup = BeautifulSoup(r.content, features="xml")
        return [item.title.text for item in soup.findAll('item')[:10]]
    except:
        return []

# --- ANALYTICS ENGINE ---
def run_analysis(headlines):
    results = []
    for text in headlines:
        clean = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        seq = tk.texts_to_sequences([clean])
        pad = pad_sequences(seq, maxlen=65)
        pred = model.predict(pad, verbose=0)
        results.append({
            "Topic": text, 
            "Sentiment": le.classes_[np.argmax(pred)], 
            "Confidence": np.max(pred)
        })
    return pd.DataFrame(results)

# --- UI LAYOUT ---
st.title("🇮🇳 Live Trend Intelligence: India vs Global")
st.markdown("This AI system performs real-time extraction and compares the **Social Pulse** of different regions.")

# TABS: The logical replacement for Historical Data
tab1, tab2 = st.tabs(["🚀 Real-Time Extraction", "⚖️ Regional Sentiment Comparison"])

with tab1:
    col_a, col_b = st.columns([1, 3])
    with col_a:
        region_choice = st.radio("Select Target Region:", ["Global", "India"])
        run_btn = st.button("Extract Top 10 Trends")
    
    if run_btn:
        headlines = fetch_trends(region_choice)
        res_df = run_analysis(headlines)
        
        with col_b:
            st.subheader(f"Current {region_choice} Top 10")
            def color_val(v):
                c = '#28a745' if v=='positive' else '#dc3545' if v=='negative' else '#ffa421'
                return f'background-color: {c}; color: white; font-weight: bold'
            st.table(res_df.style.applymap(color_val, subset=['Sentiment']))

            # New Graph: Confidence vs Topic
            fig_bar = px.bar(res_df, x='Sentiment', y='Confidence', color='Sentiment', 
                             title="Prediction Confidence Levels", template="plotly_dark")
            st.plotly_chart(fig_bar, use_container_width=True)

with tab2:
    st.subheader("Logical Comparative Analysis")
    st.write("Extracting both datasets to compare the Global mood vs the Indian mood.")
    
    if st.button("Run Comparative Intelligence"):
        with st.spinner("Syncing Global and Regional feeds..."):
            g_df = run_analysis(fetch_trends("Global"))
            i_df = run_analysis(fetch_trends("India"))

            c1, c2 = st.columns(2)
            
            # Global Chart
            with c1:
                st.write("🌍 **Global Sentiment Share**")
                fig_g = px.pie(g_df, names='Sentiment', hole=0.5, color='Sentiment',
                               color_discrete_map={'positive':'#28a745','negative':'#dc3545','neutral':'#ffa421'})
                st.plotly_chart(fig_g, use_container_width=True)
            
            # India Chart
            with c2:
                st.write("🇮🇳 **India Sentiment Share**")
                fig_i = px.pie(i_df, names='Sentiment', hole=0.5, color='Sentiment',
                               color_discrete_map={'positive':'#28a745','negative':'#dc3545','neutral':'#ffa421'})
                st.plotly_chart(fig_i, use_container_width=True)

            # THE LOGICAL CONCLUSION (AI Insight)
            st.markdown("---")
            st.subheader("📊 Strategic AI Conclusion")
            g_pos = len(g_df[g_df['Sentiment']=='positive'])
            i_pos = len(i_df[i_df['Sentiment']=='positive'])
            
            if i_pos > g_pos:
                st.success("Analysis: The **Indian Market** currently shows higher optimism than Global averages.")
            elif i_pos < g_pos:
                st.warning("Analysis: The **Indian Market** is currently more critical/cautious than Global trends.")
            else:
                st.info("Analysis: Regional and Global sentiments are currently in equilibrium.")

st.sidebar.markdown("---")
st.sidebar.info("Project: CNN-LSTM Hybrid Trend Engine")
