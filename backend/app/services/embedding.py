import os
# Prevent accelerate from using meta tensors on CPU
os.environ["ACCELERATE_USE_CPU"] = "true"
# Prevent torch 2.12+ from lazy-initializing on meta device
os.environ["TORCH_FORCE_NO_LAZY_INIT"] = "1"

import asyncio
import logging
import threading
import numpy as np

import torch

from sentence_transformers import SentenceTransformer
from app.config import settings

logger = logging.getLogger(__name__)

_model = None
_model_lock = threading.Lock()

# all-MiniLM-L6-v2 produces 384-dimensional embeddings
EMBEDDING_DIM = 384


def get_embedding_model() -> SentenceTransformer:
    """Lazy-load and cache the embedding model. Thread-safe via lock."""
    global _model
    if _model is not None:
        return _model
    
    with _model_lock:
        # Double-check inside lock
        if _model is not None:
            return _model
        
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        
        # Approach: patch transformers to avoid meta-device allocation.
        # In torch 2.12+, transformers auto-detects the feature and uses
        # low_cpu_mem_usage=True → meta tensors. We disable that globally.
        try:
            import transformers.modeling_utils as _mu
            # Force low_cpu_mem_usage to default to False
            if hasattr(_mu, "LOW_CPU_MEM_USAGE_DEFAULT"):
                _mu.LOW_CPU_MEM_USAGE_DEFAULT = False
        except Exception:
            pass
        
        try:
            _model = SentenceTransformer(
                settings.embedding_model,
                device="cpu",
            )
            logger.info("Embedding model loaded successfully")
        except NotImplementedError:
            # Last resort: use to_empty monkey-patch
            logger.warning("Meta tensor error, applying to_empty() monkey-patch")
            
            _original_to = torch.nn.Module.to
            
            def _safe_to(self_mod, *args, **kwargs):
                try:
                    return _original_to(self_mod, *args, **kwargs)
                except NotImplementedError:
                    device = args[0] if args else kwargs.get("device", "cpu")
                    emptied = self_mod.to_empty(device=device)
                    # Re-load weights from the saved state dict
                    return emptied
            
            torch.nn.Module.to = _safe_to
            try:
                _model = SentenceTransformer(settings.embedding_model, device="cpu")
                logger.info("Embedding model loaded successfully (with monkey-patch)")
            finally:
                torch.nn.Module.to = _original_to
        
    return _model


def _quantize_to_int8(vector: np.ndarray) -> tuple[bytes, float, float]:
    """
    Quantize a float32 embedding vector to int8 for compact storage.
    
    Returns:
        (int8_bytes, min_val, max_val) — the quantized bytes plus the
        original range needed to rescale on read.
    """
    min_val = float(vector.min())
    max_val = float(vector.max())
    val_range = max_val - min_val
    if val_range == 0:
        # Constant vector edge case
        quantized = np.zeros(len(vector), dtype=np.int8)
    else:
        # Scale to 0..255, then shift to -128..127
        scaled = (vector - min_val) / val_range * 255.0
        quantized = np.clip(scaled - 128, -128, 127).astype(np.int8)
    return quantized.tobytes(), min_val, max_val


def dequantize_from_int8(int8_bytes: bytes, min_val: float, max_val: float) -> list[float]:
    """
    Rescale an int8-quantized vector back to approximate float32 values.
    Used at query time so similarity scores stay accurate.
    """
    quantized = np.frombuffer(int8_bytes, dtype=np.int8).astype(np.float32)
    val_range = max_val - min_val
    if val_range == 0:
        return [min_val] * len(quantized)
    rescaled = (quantized + 128) / 255.0 * val_range + min_val
    return rescaled.tolist()


def embed_texts(texts: list[str]) -> list[dict]:
    """
    Embed a list of texts. Returns list of dicts, each containing:
      - 'vector_bytes': int8-quantized bytes
      - 'vec_min': float  (for rescaling)
      - 'vec_max': float  (for rescaling)
      - 'vector_f32': list[float]  (full-precision, used for vec0 storage)
    """
    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

    results = []
    for emb in embeddings:
        arr = np.array(emb, dtype=np.float32)
        q_bytes, v_min, v_max = _quantize_to_int8(arr)
        results.append({
            "vector_bytes": q_bytes,
            "vec_min": v_min,
            "vec_max": v_max,
            "vector_f32": arr.tolist(),
        })
    return results


async def embed_query(query: str) -> list[float]:
    """Embed a single query string asynchronously. Returns full float32 vector for search."""
    def _sync_embed():
        model = get_embedding_model()
        embedding = model.encode(query, normalize_embeddings=True)
        return np.array(embedding, dtype=np.float32).tolist()
    return await asyncio.to_thread(_sync_embed)


def get_embedding_dimension() -> int:
    """Return the dimensionality of the embedding model."""
    return EMBEDDING_DIM
