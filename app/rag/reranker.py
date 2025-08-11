from app.embedding.models import CrossEncoderReranker

cross_encoder = CrossEncoderReranker()

def rerank(query, points, docs, top_k):
    scores = [cross_encoder.compute_score(query, doc) for doc in docs]
    ranked  = sorted(zip(points, scores), key=lambda x: x[1], reverse=True)[:top_k]
    return [item for item, _ in ranked]
    