from app.db.qdrant import VectorStore
from app.embedding.embedding_models import DenseEmbedding, SparseEmbedding

dense_model = DenseEmbedding()
sparse_model = SparseEmbedding()

COLLECTIONS = {
    "stock": VectorStore(collection_name="stock_collection", dense_model=dense_model, sparse_model=sparse_model),
}