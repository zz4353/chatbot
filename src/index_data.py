import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, SparseVectorParams
from qdrant_client.models import SparseVector, HnswConfigDiff
from utils import get_files_in_directory, normalize
from langchain.schema import Document
import uuid
import torch
from fastembed import SparseTextEmbedding
from utils import preprocess_text
from collections import defaultdict
from qdrant_client.models import SparseIndexParams
from file_utils import load_documents_from_path

load_dotenv() 

QDRANT_CLIENT = QdrantClient(host=os.getenv("QDRANT_HOST"), port=os.getenv("QDRANT_PORT"), 
                             timeout=int(os.getenv("QDRANT_TIMEOUT")))
DEFAULT_COLLECTION_NAME = "local_store"
DENSE_MODEL = os.getenv("DENSE_MODEL")
SPARSE_MODEL = os.getenv("SPARSE_MODEL")

class VectorStore:
    def __init__(self, collection_name=DEFAULT_COLLECTION_NAME, dense_model_name=DENSE_MODEL, sparse_model_name=SPARSE_MODEL, device="cpu"):
        self.collection_name = collection_name
        self.device = device
        self.dense_embedding_model = SentenceTransformer(dense_model_name, cache_folder="models")
        self.dense_embedding_model.to(torch.device(self.device))
        self.sparse_embedding_model = SparseTextEmbedding(sparse_model_name, cache_dir="models",
                                                          cuda=(self.device != "cpu"))

        if not QDRANT_CLIENT.collection_exists(self.collection_name):
            self.create_collection()

    def create_collection(self):
        dense_dim = self.dense_embedding_model.encode("test").shape[0]

        QDRANT_CLIENT.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense_vector": VectorParams(
                    size=dense_dim,
                    distance=Distance.COSINE,
                )
            },

            sparse_vectors_config={
                "sparse_vector": SparseVectorParams(
                    index=SparseIndexParams()
                )
            },

            hnsw_config=HnswConfigDiff(
                m=0,
            ),
        ) 

    def enable_hnsw_indexing(self):
        QDRANT_CLIENT.update_collection(
            collection_name=self.collection_name,
            hnsw_config=HnswConfigDiff(
                m=16,
            ),
        ) 

    def get_collection_info(self):
        return QDRANT_CLIENT.get_collection(self.collection_name)

    def _embed_contents(self, chunk_contents):
        dense_embeddings = self.dense_embedding_model.encode(chunk_contents)

        chunk_contents = [preprocess_text(content) for content in chunk_contents]
        sparse_embeddings = list(self.sparse_embedding_model.passage_embed(chunk_contents))

        return dense_embeddings, sparse_embeddings
        
    def _upsert_data(self, points):
        BATCH_SIZE = 100
        for i in range(0, len(points), BATCH_SIZE):
            batch = points[i:i + BATCH_SIZE]
            QDRANT_CLIENT.upsert(
                collection_name=self.collection_name,
                points=batch
            )
            print(f"Upserted batch {i // BATCH_SIZE + 1} of {len(points) // BATCH_SIZE + 1}")


    def load_and_index_data(self, path):
        if QDRANT_CLIENT.collection_exists(self.collection_name):
            QDRANT_CLIENT.delete_collection(self.collection_name)

        self.create_collection()

        print(f"Indexing data from {path}...")
        files = get_files_in_directory(path)
        
        for path in files:
            chunk_contents = load_documents_from_path(path)

            dense_embeddings, sparse_embeddings = self._embed_contents(chunk_contents)

            self.upsert_data(
                dense_embeddings=dense_embeddings,
                sparse_embeddings=sparse_embeddings,
                payload_keys=["content"],
                payload_values=chunk_contents
            )
        
        self.enable_hnsw_indexing()

    def upsert_data(self, dense_embeddings, sparse_embeddings, payload_keys, payload_values):
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense_vector": dense_embeddings[i].tolist(),
                    "sparse_vector": sparse_embeddings[i].as_object(),
                },
                payload=dict(zip(payload_keys, payload_values[i] if isinstance(payload_values[i], list) else [payload_values[i]]))
            )
            for i in range(len(dense_embeddings))
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
        return [point for point in results if point.score >= threshold]
    
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
    
    def search_sparse(self, query, top_k=3, threshold=0.3):
        results = self._search_sparse(query, top_k)
        return [point for point in results if point.score >= threshold]

    def hybrid_search(self, query, top_k=3, threshold=0.3, alpha=0.7):
        dense_results = self._search_dense(query, top_k * 3 + 1)
        dense_scores = normalize([point.score for point in dense_results])
        for i, point in enumerate(dense_results):
            point.score = dense_scores[i]

        sparse_results = self._search_sparse(query, top_k * 3 + 1)
        sparse_scores = normalize([point.score for point in sparse_results])
        for i, point in enumerate(sparse_results):
            point.score = sparse_scores[i]

        combined = defaultdict(lambda: {"payload": None, "dense": 0.0, "sparse": 0.0})
        for p in sparse_results:
            combined[p.id]["point"] = p
            combined[p.id]["sparse"] = p.score

        for p in dense_results:
            combined[p.id]["point"] = p
            combined[p.id]["dense"] = p.score

        final = []
        for id_, entry in combined.items():
            score = alpha * entry["dense"] + (1 - alpha) * entry["sparse"]
            if score >= threshold:
                entry['point'].score = score
                final.append(entry["point"])

        final = sorted(final, key=lambda d: d.score, reverse=True)[:top_k]
        return final

    def search(self, query, top_k=3, threshold=0.3):
        docs = []
        results = self.hybrid_search(query, top_k, threshold)
        for point in results:
            docs.append(Document(
                page_content=point.payload["content"],
                metadata={"score": point.score, "id": point.id}
            ))

        return docs

if __name__ == "__main__":
    vector_store = VectorStore()
    # index_data = vector_store.load_and_index_data(path="data")
    vector_store.enable_hnsw_indexing()

    question = "chứng khoán là gì?"

    results = vector_store.hybrid_search(question, top_k=3, threshold=0.3)
    print(results)
    print("================================================================")


    results = vector_store.search_dense(query=question, top_k=3)
    print(results)

    print("================================================================")

    results = vector_store.search_sparse(query=question, top_k=3)
    print(type(results))
    print(results)

    print("================================================================")
    results = vector_store.search(query=question, top_k=3, threshold=0.3)
    print(results)
    
