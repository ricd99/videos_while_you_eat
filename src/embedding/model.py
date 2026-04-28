from functools import lru_cache
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"

@lru_cache(maxsize=1)
def get_embedding_model(model_name: str = DEFAULT_MODEL):
    """
    Load and cache a SentenceTransformer model.
    
    Uses lru_cache so the model is loaded once and reused across all
    calls. This is important because loading the model is expensive.
    
    Args:
        model_name: The name of the SentenceTransformer model to load.
                    Defaults to BAAI/bge-base-en-v1.5
    
    Returns:
        A SentenceTransformer model instance
    
    Example:
        >>> model = get_embedding_model()
        >>> embeddings = model.encode(["some text here"])
    """
    return SentenceTransformer(model_name) # autoimplements: device="cuda" if GPU, else "cpu"

def batch_encode(texts: list[str], model_name: str = DEFAULT_MODEL, normalize: bool = True):
    """
    Encode a list of text strings into embeddings.
    
    Args:
        texts: List of text strings to encode
        model_name: Which SentenceTransformer model to use
        normalize: If True, normalizes embeddings to unit length (important for cosine similarity)
    
    Returns:
        numpy array of shape (len(texts), embedding_dim)
    
    Example:
        >>> texts = ["channel about movies", "cooking tutorial"]
        >>> embeddings = encode_texts(texts)
        >>> print(embeddings.shape)  # (2, 768) or similar
    """
    model = get_embedding_model(model_name)
    return model.encode(texts, normalize_embeddings=normalize, show_progress_bar=True)

def encode(text: str, model_name: str = DEFAULT_MODEL, normalize: bool = True):
    """
    Encode a single text string into an embedding.
    
    Args:
        text: A single text string
        model_name: Which SentenceTransformer model to use
        normalize: If True, normalizes to unit length
    
    Returns:
        numpy array of shape (embedding_dim,)
    
    Example:
        >>> embedding = encode_single("a video essay channel about film")
        >>> print(embedding.shape)  # (768,)
    """
    model = get_embedding_model(model_name)
    return model.encode(text, normalize_embeddings=normalize)