import os
import requests
from typing import List, Tuple
from abc import ABC, abstractmethod
import chromadb
from langchain.embeddings.base import Embeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

# ------------------ Embedding Strategy ------------------
class EmbeddingInterface(ABC):
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        pass

class JinaAIEmbeddings(Embeddings, EmbeddingInterface):
    def __init__(self, api_key: str, model_name: str = "jina-embeddings-v3"):
        self.api_key = api_key
        self.model_name = model_name
        self.api_url = "https://api.jina.ai/v1/embeddings"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        payload = {
            "model": self.model_name,
            "task": "retrieval.passage",
            "input": texts
        }
        response = requests.post(self.api_url, headers=self.headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return [item["embedding"] for item in data["data"]]

    def embed_query(self, text: str) -> List[float]:
        payload = {
            "model": self.model_name,
            "task": "retrieval.query",
            "input": [text]
        }
        response = requests.post(self.api_url, headers=self.headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["data"][0]["embedding"]

# ------------------ Document Loader Factory ------------------
class DocumentLoaderFactory:
    @staticmethod
    def get_loader(file_path: str):
        if file_path.lower().endswith(".pdf"):
            return PyPDFLoader(file_path)
        elif file_path.lower().endswith(".txt"):
            return TextLoader(file_path, encoding="utf8")
        else:
            raise ValueError("❌ Unsupported file type. Only .txt and .pdf are supported.")

# ------------------ Document Processor ------------------
class DocumentProcessor:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def process(self, loader) -> Tuple[List[str], List[dict], List[str]]:
        documents = loader.load()
        docs = self.splitter.split_documents(documents)
        texts = [doc.page_content for doc in docs]
        metadatas = [doc.metadata for doc in docs]
        ids = [f"doc_{i}" for i in range(len(texts))]
        return texts, metadatas, ids

# ------------------ ChromaDB Store ------------------
class ChromaDBStore:
    def __init__(self):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(name='documents')

    def add_documents(self, texts: List[str], metadatas: List[dict], ids: List[str], embeddings: List[List[float]]):
        self.collection.add(documents=texts, metadatas=metadatas, ids=ids, embeddings=embeddings)

    def query_documents(self, query_embedding: List[float], n_results: int = 3):
        return self.collection.query(query_embeddings=[query_embedding], n_results=n_results)

# ------------------ User Interaction ------------------
class UserInterface:
    @staticmethod
    def choose_file() -> str:
        current_dir = os.path.dirname(__file__)
        files = {
            str(i + 1): f for i, f in enumerate(os.listdir(current_dir))
            if os.path.isfile(os.path.join(current_dir, f))
        }

        print("\n📄 Available Documents:")
        for i, f in files.items():
            print(f"{i}. {f}")

        chosen_id = input("\n📝 Enter the file ID you want to embed: ").strip()
        file_name = files.get(chosen_id)
        if not file_name:
            raise FileNotFoundError("❌ File not found.")
        return os.path.join(current_dir, file_name)

    @staticmethod
    def get_query() -> str:
        return input("\n❓ Ask something from your docs: ").strip()

    @staticmethod
    def display_results(results):
        print("\n🔍 Top Results:")
        for i, doc in enumerate(results['documents'][0]):
            print(f"\nResult {i + 1}:\n{doc}")

# ------------------ Application Controller ------------------
# Controls the workflow of the application
class AppController:
    def __init__(self, embedding_service: EmbeddingInterface):
        self.embedding_service = embedding_service
        self.processor = DocumentProcessor()
        self.store = ChromaDBStore()

    def run(self):
        while True:
            try:
                print("\n🔁 New Session (type 'q' at any time to quit)")

                # Step 1: File selection
                file_path = UserInterface.choose_file()
                if file_path.lower().strip() == 'q':
                    print("👋 Exiting the app. Goodbye!")
                    break

                # Step 2: Load and process file
                loader = DocumentLoaderFactory.get_loader(file_path)
                texts, metadatas, ids = self.processor.process(loader)

                # Step 3: Embed and store
                embeddings = self.embedding_service.embed_documents(texts)
                self.store.add_documents(texts, metadatas, ids, embeddings)
                print("✅ Successfully embedded and stored in ChromaDB!")

                # Step 4: Ask questions in a loop
                while True:
                    query = UserInterface.get_query()
                    if query.lower().strip() == 'q':
                        print("📁 Done with this document.")
                        break

                    query_embedding = self.embedding_service.embed_query(query)
                    results = self.store.query_documents(query_embedding)
                    UserInterface.display_results(results)

            except Exception as e:
                print(f"⚠️ Error: {e}")


# ------------------ Main Entry ------------------
def main():
    api_key = os.getenv("JINA_API_KEY")
    if not api_key:
        raise ValueError("❌ JINA_API_KEY not found in .env file.")

    embedding_service = JinaAIEmbeddings(api_key=api_key)
    app = AppController(embedding_service)
    app.run()

if __name__ == "__main__":
    main()
