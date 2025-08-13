from app.db.qdrant import VectorStore
from app.embedding.models import DenseEmbedding, SparseEmbedding

dense_model = DenseEmbedding()
sparse_model1 = SparseEmbedding()  # mỗi collection tạo 1 sparse model riêng. 

COLLECTIONS = {
    "stock": VectorStore(collection_name="stock_collection", dense_model=dense_model, sparse_model=sparse_model1),
}