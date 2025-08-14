import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import streamlit as st
from pipeline.pipeline import AnimeRecommendationPipeline
from dotenv import load_dotenv


st.set_page_config(
    page_title="Anime Recommendation System",
    layout="wide",
)

load_dotenv()

@st.cache_resource
def initialize_pipeline():
    return AnimeRecommendationPipeline(persist_dir='../chroma_db')

pipeline = initialize_pipeline()

st.title("Anime Recommendation System")

query = st.text_input("Enter your anime prefernces eg. : light hearted anime with school settings")
if query:
    with st.spinner("Fetching recommendations for you....."):
        response = pipeline.recommend(query)
        st.markdown("### Recommendations")
        st.write(response)
