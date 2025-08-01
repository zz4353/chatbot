import os
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from utils import convert_to_markdown, get_files_in_directory, has_hash_changed
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


class Vector_Store:
    def __init__(self, embedding_model_name="intfloat/multilingual-e5-base"):
        local_path = os.path.join("models", embedding_model_name.replace("/", "_"))

        if os.path.exists(local_path):
            print("Loading embedding model from local path...")
            self.embedding_model = SentenceTransformer(local_path)
        else:
            print("Downloading embedding model...")
            self.embedding_model = SentenceTransformer(embedding_model_name)
            self.embedding_model.save(local_path)

        self.qdrant_client = QdrantClient(host="127.0.0.1", port=6333, timeout=6000)
        self.collection_name = "local_store"

    def _embed_contents(self, chunk_contents):
        # Chia nhỏ dữ liệu, up nhiều dễ nhìn 
        BATCH_SIZE = 100
        embeddings = []

        for i in range(0, len(chunk_contents), BATCH_SIZE):
            batch_contents = chunk_contents[i:i + BATCH_SIZE]
            batch_embeddings = self.embedding_model.encode(batch_contents)
            embeddings.extend(batch_embeddings)
            print(f"Processed batch {i // BATCH_SIZE + 1} of {len(chunk_contents) // BATCH_SIZE + 1}")
        return embeddings
        
    def _upsert_data(self, points):
        # Chia nhỏ dữ liệu, up nhiều bị lỗi 
        BATCH_SIZE = 100
        for i in range(0, len(points), BATCH_SIZE):
            batch = points[i:i + BATCH_SIZE]
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=batch
            )
            print(f"Upserted batch {i // BATCH_SIZE + 1} of {len(points) // BATCH_SIZE + 1}")


    def index_data(self, path):
        # Kiểm tra xem có cần cập nhật không
        if not has_hash_changed(path):
            print("Data has not changed, skipping indexing.")
            return

        print(f"Indexing data from {path}...")

        # Lấy số chiều của vector từ mô hình embedding
        sample_vector = self.embedding_model.encode("test") 
        dim = sample_vector.shape[0]

        # Tạo mới collection
        self.qdrant_client.recreate_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                )

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                    chunk_size=512, chunk_overlap=126
                )
        
        idx = 0 
        files = get_files_in_directory(path)
        
        for f in files:
            file_name = f.split("/")[-1]
            chunk_contents = []
            # Xử lý file .docx, .txt và .pdf
            if f.endswith(".docx") or f.endswith(".txt") or f.endswith(".pdf"):
                print(f"Processing file: {f}")
                markdown_text = convert_to_markdown(f)
                docs = [Document(page_content=markdown_text)]
                chunks = text_splitter.transform_documents(docs)
                chunk_contents = ["Nguồn: " + file_name + '\n' + doc.page_content.replace("\n", " ") for doc in chunks if len(doc.page_content) >= 40]

            # Xử lý file .json
            elif f.endswith(".json"):
                print(f"Processing file: {f}")
                with open(f, "r", encoding="utf-8") as json_file:
                    data = json.load(json_file)

                for entry in data:
                    content = entry.get("content", "")
                    metadata = entry.get("metadata", "")

                    if not content.strip():
                        continue

                    docs = [Document(page_content=content)]
                    chunks = text_splitter.transform_documents(docs)

                    # Gắn metadata vào mỗi chunk
                    for doc in chunks:
                        if len(doc.page_content) >= 40:
                            chunk_contents.append(f"[{metadata}]\n{doc.page_content}")

            else:
                # Bỏ qua các file khác
                continue
            
            # embedding 
            embeddings = self._embed_contents(chunk_contents)

            # Tạo danh sách PointStruct để upsert vào Qdrant
            points = [
                PointStruct(
                    id= i + idx,
                    vector=embeddings[i].tolist(),
                    payload={"text": chunk_contents[i]},
                )
                for i in range(len(chunk_contents))
            ]
            idx += len(chunk_contents)

            # up dữ liệu vào Qdrant
            self._upsert_data(points)

    def search(self, query, top_k=5):
        query_vector = self.embedding_model.encode([query])[0]

        hits = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_vector.tolist(),
            limit=top_k
        )

        return [Document(page_content=hit.payload["text"], metadata={"score": hit.score}) for hit in hits]


if __name__ == "__main__":
    vector_store = Vector_Store()
    index_data = vector_store.index_data(path="data")
    context = vector_store.search(query="Lý do gì khiến SSI giữ nguyên dự báo lợi nhuận giai đoạn 2024–2028?", top_k = 3)
    for doc in context:
        print(doc.page_content, doc.metadata)

