import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import plotly.express as px

# PAGE CONFIG
st.set_page_config(page_title="Big Data Social Media Trends", layout="wide")

# CUSTOM CSS
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    </style>
    """, unsafe_allow_html=True)

# LOAD ASSETS
@st.cache_resource
def load_assets():
    model = load_model('sentiment_model.h5')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    with open('label_encoder.pickle', 'rb') as handle:
        le = pickle.load(handle)
    df = pd.read_csv('test.csv')
    return model, tokenizer, le, df

model, tokenizer, le, raw_df = load_assets()

# SIDEBAR
st.sidebar.title("Project Controls")
st.sidebar.info("This project analyzes Big Data from Social Media to identify global trends and sentiments.")

# MAIN TITLE
st.title("📊 Big Data Analysis: Social Media Trends")
st.markdown("---")

# TABS FOR NAVIGATION
tab1, tab2 = st.tabs(["🔍 Sentiment Predictor", "📈 Trend Dashboard"])

with tab1:
    st.header("Real-time Sentiment Prediction")
    user_tweet = st.text_area("Enter a social media post/tweet below:", "I am feeling so excited about the new technology trends!")
    
    if st.button("Analyze Sentiment"):
        # Preprocessing (Simplified)
        clean_text = user_tweet.lower()
        clean_text = re.sub(r'[^\w\s]', '', clean_text)
        
        # Vectorize
        seq = tokenizer.texts_to_sequences([clean_text])
        padded = pad_sequences(seq, maxlen=50) # Matching Training config
        
        # Predict
        pred = model.predict(padded)
        labels = le.classes_
        result = labels[np.argmax(pred)]
        confidence = np.max(pred) * 100
        
        # Display Output
        col1, col2 = st.columns(2)
        with col1:
            if result == 'positive':
                st.success(f"Detected Sentiment: **{result.upper()}**")
            elif result == 'negative':
                st.error(f"Detected Sentiment: **{result.upper()}**")
            else:
                st.warning(f"Detected Sentiment: **{result.upper()}**")
        
        with col2:
            st.metric("Confidence Score", f"{confidence:.2f}%")

with tab2:
    st.header("Global Social Media Trends")
    
    # Trend 1: Sentiment by Age
    st.subheader("1. Sentiment Distribution by Age Group")
    fig_age = px.histogram(raw_df, x="Age of User", color="sentiment", barmode="group", 
                           color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig_age, use_container_width=True)
    
    # Trend 2: Time of Day Analysis
    st.subheader("2. Impact of Time on User Sentiment")
    fig_time = px.pie(raw_df, names='Time of Tweet', values='Population -2020', 
                      title="Tweet Volume by Time of Day", hole=0.4)
    st.plotly_chart(fig_time, use_container_width=True)

    # Trend 3: Country Map
    st.subheader("3. Geographic Trend Map")
    fig_map = px.choropleth(raw_df, locations="Country", locationmode='country names',
                            color="sentiment", hover_name="Country",
                            title="Primary Sentiment by Country")
    st.plotly_chart(fig_map, use_container_width=True)

st.markdown("---")
st.caption("College Project: Big Data Analysis Framework | Built with TensorFlow & Streamlit")
