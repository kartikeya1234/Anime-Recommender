from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv
load_dotenv()


class VectorStoreBuilder:
    def __init__(self, csv_path: str, persistent_dir:str = "chroma_db") -> None:
        self.csv_path = csv_path
        self.persistent_dir = persistent_dir
        self.embedding = OpenAIEmbeddings(model='text-embedding-3-large')

    def build_and_save_vectorstore(self):
        loader = CSVLoader(
            file_path=self.csv_path,
            encoding='utf-8',
            metadata_columns=[]
        )

        data = loader.load()

        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = splitter.split_documents(data)

        db = Chroma.from_documents(texts, embedding=self.embedding, persist_directory=self.persistent_dir)
        db.persist()

    def load_vector_store(self):
        return Chroma(persist_directory=self.persistent_dir)
