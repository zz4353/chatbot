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

        if not any(c.name == collection_name for c in QDRANT_CLIENT.get_collections().collections):
            self.recreate_collection()

    def recreate_collection(self):
        dense_dim = self.dense_embedding_model.encode("test").shape[0]

        QDRANT_CLIENT.recreate_collection(
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


    def index_data(self, path):
        self.recreate_collection()

        print(f"Indexing data from {path}...")
        files = get_files_in_directory(path)
        
        for path in files:
            chunk_contents = load_documents_from_path(path)
            
            # embedding 
            dense_embeddings, sparse_embeddings = self._embed_contents(chunk_contents)

            # Tạo danh sách PointStruct để upsert vào Qdrant
            points = [
                PointStruct(
                    id= str(uuid.uuid4()),
                    vector={
                        "dense_vector" : dense_embeddings[i].tolist(),
                        "sparse_vector" : sparse_embeddings[i].as_object(),
                    },
                    payload={"text": chunk_contents[i]},
                )
                for i in range(len(chunk_contents))
            ]

            self._upsert_data(points)
        
        self.enable_hnsw_indexing()

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
    vector_store.enable_hnsw_indexing()
    # index_data = vector_store.index_data(path="data")

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

    
