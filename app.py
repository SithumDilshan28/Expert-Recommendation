import streamlit as st
import pickle
import pandas as pd
from PIL import Image
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load the model and TF-IDF vectorizer from pickle files
with open('expert_recommendation_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as tfidf_file:
    tfidf = pickle.load(tfidf_file)

# Set page layout and title
st.set_page_config(page_title="Expert Recommendation", layout="wide", page_icon="🤖")
st.title("🔍 Expert Recommendation System")
st.subheader("Get the right expert for your problem in seconds!")

# Sidebar with additional options
with st.sidebar:
    st.image("expert_banner.png", use_column_width=True)  # You can use a custom image here
    st.header("About")
    st.write(
        """
        This app uses Machine Learning and NLP to recommend experts based on your problem description. 
        Simply describe your issue, and get matched with an expert in Legal, Fitness, Career, Health, Finance, or Relationships.
        """
    )
    st.write("Developed by: U-Connect")

# Input from the user
st.markdown("### Describe your problem:")
user_input = st.text_area("Enter the description of your problem here", height=150)

# Predict expert category on button click
if st.button("Get Expert Recommendation"):
    if user_input.strip():
        # Preprocess the input and make predictions
        user_input_transformed = tfidf.transform([user_input])
        predicted_expert = model.predict(user_input_transformed)

        st.success(f"**Recommended Expert: {predicted_expert[0]}**")
        
        # Show a visual representation of the input text
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(user_input)
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(plt)

    else:
        st.error("Please enter a valid problem description!")

# Add footer with social media links
st.markdown("---")
st.markdown("💬 Connect with us: [LinkedIn](https://www.linkedin.com/) | [Twitter](https://twitter.com/) | [GitHub](https://github.com/)")

# Style adjustments using HTML/CSS
st.markdown("""
    <style>
    body {
        background-color: #f0f2f6;
        color: #333;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        padding: 10px 24px;
        border-radius: 10px;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    </style>
    """, unsafe_allow_html=True)
