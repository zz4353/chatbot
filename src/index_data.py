import os
import json
from typing import final
import pandas as pd
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, SparseVectorParams, Modifier, Prefetch
from qdrant_client.models import FusionQuery, Fusion, SparseVector
from utils import convert_to_markdown, get_files_in_directory, has_hash_changed, normalize
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import uuid
from fastembed import SparseTextEmbedding
from utils import preprocess_text
from collections import defaultdict

QDRANT_CLIENT = QdrantClient(host="127.0.0.1", port=6333, timeout=6000)

class VectorStore:
    def __init__(self, collection_name="local_store", embedding_model_name="intfloat/multilingual-e5-base"):
        local_path = os.path.join("models", embedding_model_name.replace("/", "_"))

        if os.path.exists(local_path):
            print("Loading embedding model from local path...")
            self.dense_embedding_model = SentenceTransformer(local_path)
        else:
            print("Downloading embedding model...")
            self.dense_embedding_model = SentenceTransformer(embedding_model_name)
            self.dense_embedding_model.save(local_path)
        self.sparse_embedding_model = SparseTextEmbedding("Qdrant/bm25")
        self.collection_name = collection_name

    def _embed_contents(self, chunk_contents):
        # Chia nhỏ dữ liệu, up nhiều dễ nhìn 
        BATCH_SIZE = 100
        embeddings = []
        sparse_embeddings = []

        for i in range(0, len(chunk_contents), BATCH_SIZE):
            batch_contents = chunk_contents[i:i + BATCH_SIZE]
            batch_embeddings = self.dense_embedding_model.encode(batch_contents)
            embeddings.extend(batch_embeddings)

            batch_contents = [preprocess_text(content) for content in batch_contents]
            bm25_embeddings = list(self.sparse_embedding_model.passage_embed(batch_contents))
            sparse_embeddings.extend(bm25_embeddings)

            print(f"Processed batch {i // BATCH_SIZE + 1} of {len(chunk_contents) // BATCH_SIZE + 1}")
        return embeddings, sparse_embeddings
        
    def _upsert_data(self, points):
        # Chia nhỏ dữ liệu, up nhiều bị lỗi 
        BATCH_SIZE = 100
        for i in range(0, len(points), BATCH_SIZE):
            batch = points[i:i + BATCH_SIZE]
            QDRANT_CLIENT.upsert(
                collection_name=self.collection_name,
                points=batch
            )
            print(f"Upserted batch {i // BATCH_SIZE + 1} of {len(points) // BATCH_SIZE + 1}")

    def recreate_collection(self, dim):
        QDRANT_CLIENT.recreate_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense_vector": VectorParams(
                    size=dim,
                    distance=Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "sparse_vector": SparseVectorParams(
                    modifier=Modifier.IDF,
                )
            }
        )

    def index_data(self, path):
        # Kiểm tra xem có cần cập nhật không
        if not has_hash_changed(path):
            print("Data has not changed, skipping indexing.")
            return

        print(f"Indexing data from {path}...")

        # Tạo mới collection
        self.recreate_collection(self.dense_embedding_model.encode("test").shape[0])

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                    chunk_size=512, chunk_overlap=126
                )
        
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
            embeddings, sparse_embeddings = self._embed_contents(chunk_contents)

            # Tạo danh sách PointStruct để upsert vào Qdrant
            points = [
                PointStruct(
                    id= str(uuid.uuid4()),
                    vector={
                        "dense_vector" : embeddings[i].tolist(),
                        "sparse_vector" : sparse_embeddings[i].as_object(),
                    },
                    payload={"text": chunk_contents[i]},
                )
                for i in range(len(chunk_contents))
            ]

            self._upsert_data(points)

    def _search_dense(self, query, top_k):
        query_vector = self.dense_embedding_model.encode([query])[0]
    
        results = QDRANT_CLIENT.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist(),  
            using="dense_vector",  
            with_payload=True,
            limit=top_k,
        )
        
        return results.points 
    
    def search_dense(self, query, top_k=3, threshold=0.3):
        results = self._search_dense(query, top_k)
        final_results = []
        for point in results:
            if point.score >= threshold:
                final_results.append(Document(
                    page_content=point.payload["text"],
                    metadata={"score": point.score, "id": point.id}
                ))
        return sorted(final_results, key=lambda d: d.metadata["score"], reverse=True)[:top_k]
    
    def _search_sparse(self, query, top_k):
        query = preprocess_text(query)
        sparse_vector_dict = next(self.sparse_embedding_model.passage_embed([query])).as_object()

        sparse_vector = SparseVector(
            indices=sparse_vector_dict["indices"],
            values=sparse_vector_dict["values"]
        )

        results = QDRANT_CLIENT.query_points(
            collection_name=self.collection_name,
            query=sparse_vector,  
            using="sparse_vector",  
            with_payload=True,
            limit=top_k,
        )
        return results.points


    def _hybrid_search(self, query, top_k=3, threshold=0.3, alpha=0.7):
        dense_results = self._search_dense(query, top_k * 3)
        dense_scores = normalize([point.score for point in dense_results])
        for i, point in enumerate(dense_results):
            point.score = dense_scores[i]

        sparse_results = self._search_sparse(query, top_k * 3)
        sparse_scores = normalize([point.score for point in sparse_results])
        for i, point in enumerate(sparse_results):
            point.score = sparse_scores[i]

        combined = defaultdict(lambda: {"text": "", "dense": 0.0, "sparse": 0.0})
        for p in dense_results:
            combined[p.id]["text"] = p.payload["text"]
            combined[p.id]["dense"] = p.score

        for p in sparse_results:
            combined[p.id]["text"] = p.payload["text"]
            combined[p.id]["sparse"] = p.score

        final = []
        for id_, entry in combined.items():
            score = alpha * entry["dense"] + (1 - alpha) * entry["sparse"]
            if score >= threshold:
                final.append(Document(
                    page_content=entry["text"],
                    metadata={"score": score, "id": id_}
                ))

        final = sorted(final, key=lambda d: d.metadata["score"], reverse=True)[:top_k]
        return final

    def search(self, query, top_k=3, threshold=0.3):
        return self._hybrid_search(query, top_k, threshold)


if __name__ == "__main__":
    vector_store = VectorStore()
    index_data = vector_store.index_data(path="data")

    question = "chứng khoán là gì?"

    # results = vector_store.search(query=question, top_k=3)
    # for p in results:
    #     print(p)
    results = vector_store._hybrid_search(question, top_k=3, threshold=0.3)
    print(results)
    print("================================================================")


    results = vector_store._search_dense(query=question, top_k=3)
    print(results)

    print("================================================================")

    results = vector_store._search_sparse(query=question, top_k=3)
    print(type(results))
    print(results)
