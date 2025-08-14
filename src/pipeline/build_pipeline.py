import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data_loader import AnimeDataLoader
from src.vector_store import VectorStoreBuilder
from dotenv import load_dotenv
from utils.logger import get_logger
from utils.custom_exception import CustomException

load_dotenv()

logger = get_logger(__name__)


def main():
    try:
        logger.info("Starting to build pipeline ...")

        loader = AnimeDataLoader("data/anime_with_synopsis.csv", "data/anime_updated.csv")
        processed_csv = loader.load_and_process()

        logger.info("Data loaded and processed ...")

        vector_builder = VectorStoreBuilder(processed_csv)
        vector_builder.build_and_save_vectorstore()

        logger.info("Vector store built succesfully ...")
        logger.info("Pipeline built succesfully")

    except Exception as e:
        logger.error(f"Failed to intialize pipeline {str(e)}")
        raise CustomException("Error during pipeline intialization", e)

if __name__ == '__main__':
    main()
