from vector_store import VectorStoreBuilder
from recommender import AnimeRecommender
from config.config import OPENAI_API_KEY, MODEL_NAME
from utils.logger import get_logger
from utils.custom_exception import CustomException


logger = get_logger(__name__)


class AnimeRecommendationPipeline:
    def __init__(self, persist_dir="chroma_db") -> None:
        try:
            logger.info("Intialising Recommendation Pipeline")

            vector_builder = VectorStoreBuilder(csv_path="", persistent_dir=persist_dir)
            retriever = vector_builder.load_vector_store().as_retriever()

            self.recommender = AnimeRecommender(
                retriever=retriever,
                model_name=MODEL_NAME,
                api_key=OPENAI_API_KEY
            )

            logger.info("Pipeline intialized successfully...")

        except Exception as e:
            logger.error(f"Failed to intialize pipeline {str(e)}")
            raise CustomException("Error during pipeline intialization", e)

    def recommend(self, query:str):
        try:
            logger.info(f"Received a query {query}")

            recommendation = self.recommender.get_recommendation(query=query)

            logger.info("Recommendation Generated succesfully ...")
            return recommendation
        except Exception as e:
            logger.error(f"Failed to process the query {str(e)}")
            raise CustomException("Error during query processing", e)
