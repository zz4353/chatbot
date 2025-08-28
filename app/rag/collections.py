from app.db.qdrant import VectorStore
from app.embedding.models import DenseEmbedding, SparseEmbedding, CrossEncoderReranker

dense_model = DenseEmbedding()
sparse_model = SparseEmbedding()  
cross_encoder = CrossEncoderReranker()

COLLECTIONS = {
    "stock": VectorStore(collection_name="stock_collection", dense_model=dense_model, sparse_model=sparse_model, cross_encoder=cross_encoder),
}