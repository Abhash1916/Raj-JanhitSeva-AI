"""Small in-process replacement for Chroma + a hosted embedding API.

Why: the deploy target (Vercel) caps a function at 500 MB and has a read-only
filesystem. Chroma (+ its deps) is ~250 MB and needs a writable SQLite file;
sentence-transformers pulls in PyTorch (GBs); the hosted Hugging Face API
needs paid credits. The corpus is tiny (157 schemes -> ~5.2k vectors x 384
dims, ~8 MB), so an exact NumPy search is both simpler and faster than HNSW.

- LocalEmbedder runs the SAME model the index was built with
  (paraphrase-multilingual-MiniLM-L12-v2) as an 8-bit ONNX export through
  onnxruntime: mean pooling, no normalization - identical recipe to
  sentence-transformers. Verified against the stored fp32 vectors: cosine
  ~0.994, same distance scale, so rag_service's distance-tuned boosts hold.
- VectorCollection exposes exactly the slice of the Chroma/LangChain API that
  rag_service uses: get(where=..., include=...) and
  similarity_search_with_score(query, k). Scores are squared L2 distance,
  same as Chroma's default "l2" space (lower = more similar).

Data layout (backend/data/):
  embedding_model/model.onnx.part*   - ONNX model, split to stay under
                                       GitHub's 100 MB per-file limit
  embedding_model/tokenizer.json
  vector_index/<collection>.json     - {"ids", "documents", "metadatas"}
  vector_index/<collection>.npy      - float32 [n, 384], same row order
"""

import json
import threading
from pathlib import Path

import numpy as np
import onnxruntime as ort
from langchain_core.documents import Document
from tokenizers import Tokenizer

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
MODEL_DIR = DATA_DIR / "embedding_model"
INDEX_DIR = DATA_DIR / "vector_index"
MAX_SEQ_LENGTH = 128  # the model's own max_seq_length


class LocalEmbedder:
    def __init__(self, model_dir: Path = MODEL_DIR):
        parts = sorted(model_dir.glob("model.onnx.part*"))
        if not parts:
            raise RuntimeError(f"Embedding model not found in {model_dir}")
        model_bytes = b"".join(p.read_bytes() for p in parts)
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self._session = ort.InferenceSession(model_bytes, sess_options=opts,
                                             providers=["CPUExecutionProvider"])
        self._input_names = {i.name for i in self._session.get_inputs()}
        self._tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self._tokenizer.enable_truncation(max_length=MAX_SEQ_LENGTH)
        self._tokenizer.enable_padding(pad_id=1, pad_token="<pad>")
        self._lock = threading.Lock()  # tokenizer padding config isn't thread-safe to share

    def embed(self, texts: list[str]) -> np.ndarray:
        with self._lock:
            encodings = self._tokenizer.encode_batch(texts)
        ids = np.array([e.ids for e in encodings], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.zeros_like(ids)
        token_vectors = self._session.run(None, feeds)[0]
        weights = mask[..., None].astype(np.float32)
        return (token_vectors * weights).sum(axis=1) / np.clip(weights.sum(axis=1), 1e-9, None)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


def _matches(meta: dict, where: dict) -> bool:
    """Supports the Chroma `where` shapes rag_service uses: {"$and": [...]},
    {"field": value} and {"field": {"$in": [...]}}."""
    for key, cond in where.items():
        if key == "$and":
            if not all(_matches(meta, c) for c in cond):
                return False
        elif isinstance(cond, dict):
            if "$in" in cond:
                if meta.get(key) not in cond["$in"]:
                    return False
            else:
                raise ValueError(f"Unsupported where operator: {cond}")
        elif meta.get(key) != cond:
            return False
    return True


class VectorCollection:
    def __init__(self, name: str, embedder: LocalEmbedder, index_dir: Path = INDEX_DIR):
        with open(index_dir / f"{name}.json", encoding="utf-8") as f:
            data = json.load(f)
        self.ids = data["ids"]
        self.documents = data["documents"]
        self.metadatas = data["metadatas"]
        self.vectors = np.load(index_dir / f"{name}.npy").astype(np.float32)
        if not (len(self.ids) == len(self.documents) == len(self.metadatas) == len(self.vectors)):
            raise RuntimeError(f"Vector index '{name}' is inconsistent")
        self._embedder = embedder

    def count(self) -> int:
        return len(self.ids)

    def get(self, where: dict | None = None, include=("documents", "metadatas"), limit: int | None = None):
        rows = [i for i, m in enumerate(self.metadatas) if where is None or _matches(m, where)]
        if limit is not None:
            rows = rows[:limit]
        out = {"ids": [self.ids[i] for i in rows]}
        if "documents" in include:
            out["documents"] = [self.documents[i] for i in rows]
        if "metadatas" in include:
            out["metadatas"] = [self.metadatas[i] for i in rows]
        if "embeddings" in include:
            out["embeddings"] = self.vectors[rows]
        return out

    def similarity_search(self, query: str, k: int = 4):
        return [doc for doc, _ in self.similarity_search_with_score(query, k)]

    def similarity_search_with_score(self, query: str, k: int = 4):
        q = self._embedder.embed_query(query)
        distances = ((self.vectors - q) ** 2).sum(axis=1)  # squared L2, like Chroma's "l2"
        k = min(k, len(distances))
        top = np.argpartition(distances, k - 1)[:k]
        top = top[np.argsort(distances[top])]
        return [
            (Document(page_content=self.documents[i], metadata=self.metadatas[i]), float(distances[i]))
            for i in top
        ]


def write_collection(name: str, documents: list, embedder: LocalEmbedder,
                     index_dir: Path = INDEX_DIR, batch_size: int = 64) -> int:
    """Embed LangChain Documents and write <name>.json + <name>.npy (used by
    scripts/pipeline/build_knowledge_base.py). Overwrites any previous build."""
    index_dir.mkdir(parents=True, exist_ok=True)
    texts = [d.page_content for d in documents]
    vectors = np.concatenate(
        [embedder.embed(texts[i:i + batch_size]) for i in range(0, len(texts), batch_size)]
    ).astype(np.float32) if texts else np.zeros((0, 384), dtype=np.float32)
    with open(index_dir / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump({"ids": [f"{name}-{i}" for i in range(len(documents))],
                   "documents": texts,
                   "metadatas": [d.metadata for d in documents]},
                  f, ensure_ascii=False, separators=(",", ":"))
    np.save(index_dir / f"{name}.npy", vectors)
    return len(documents)
