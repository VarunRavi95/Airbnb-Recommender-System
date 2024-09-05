import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load precomputed data
df = pd.read_csv('processed_listings_with_original_descriptions.csv')
listings_embeddings = np.load('listings_embeddings.npy')

# Cache the model loading
@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Load the model once
model = load_model()

# Define recommendation function
def get_recommendations(query, embeddings, df, top_n=20):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)
    top_indices = similarities[0].argsort()[-top_n:][::-1]
    return df.iloc[top_indices]

# Streamlit interface
st.title("Real Estate Listing Recommender")

# User query input
query = st.text_input("Enter your search query:")

if query:
    # Get recommendations
    recommendations = get_recommendations(query, listings_embeddings, df)

    # Display results
    for idx, row in recommendations.iterrows():
        col1, col2 = st.columns([1, 2])  # Set column layout with different widths
        with col1:
            # Use Markdown to make the image clickable
            st.markdown(
                f'<a href="{row["listing_url"]}" target="_blank">'
                f'<img src="{row["picture_url"]}" width="200"></a>',
                unsafe_allow_html=True
            )
        with col2:
            st.write(f"**Description:**\n {row['description']}")  # Display the description
            #st.write("\n")  # Line break
            st.write(f"**Neighborhood:** {row['neighbourhood']}")  # Display the neighborhood after a line break