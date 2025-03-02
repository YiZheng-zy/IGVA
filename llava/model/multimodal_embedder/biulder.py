from sentence_transformers import SentenceTransformer

def build_sentence_embedder(model_path):
    return SentenceTransformer(model_path)
    