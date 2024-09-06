import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import streamlit as st

# Load precomputed data
df = pd.read_csv('processed_listings_with_original_descriptions.csv')
listings_embeddings = np.load('listings_embeddings.npy').astype('float32')  # FAISS requires float32

# Cache the model loading
@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Load the model once
model = load_model()

# Create a FAISS index
index = faiss.IndexFlatL2(listings_embeddings.shape[1])  # L2 (euclidean) distance
index.add(listings_embeddings)

# Define recommendation function using FAISS
def get_recommendations(query, index, df, top_n=20):
    query_embedding = model.encode([query]).astype('float32')  # Convert to float32 for FAISS
    distances, top_indices = index.search(query_embedding, top_n)  # Search in the index
    return df.iloc[top_indices[0]]

# Streamlit interface
st.title("Real Estate Listing Recommender")

# User query input
query = st.text_input("Enter your search query:")

if query:
    # Get recommendations
    recommendations = get_recommendations(query, index, df)

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
            st.write(f"**Neighborhood:** {row['neighbourhood']}")  # Display the neighborhood