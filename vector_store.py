"""
vector_store.py — Phase 3: Qdrant Vector Database (Local + Server Mode)

Supports two deployment modes controlled by config.QDRANT_MODE:
  "local"  → file-based Qdrant (single-user, no server needed)
  "server" → connects to a running Qdrant server (multi-user, production)

All vectors are L2-normalised before upload so cosine similarity
matches Qdrant's COSINE distance metric.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer
from rich.progress import track

import config
from utils import console, get_logger

if TYPE_CHECKING:
    from document_loader import Document

logger = get_logger(__name__)


class VectorStore:
    """
    Qdrant-backed semantic retrieval store with local/server mode support.
    """

    def __init__(self) -> None:
        console.print("[cyan]⚡ Initialising Vector Store…[/cyan]")

        # Embedding model
        self.embed_model = SentenceTransformer(config.EMBEDDING_MODEL)
        console.print(
            f"  [green]✓[/green] Embedding model: [bold]{config.EMBEDDING_MODEL}[/bold] "
            f"({config.EMBEDDING_DIM}d)"
        )

        # Qdrant client — local or server mode
        if config.QDRANT_MODE == "server":
            self.client = QdrantClient(
                host=config.QDRANT_HOST,
                port=config.QDRANT_PORT,
            )
            console.print(
                f"  [green]✓[/green] Qdrant server: "
                f"[bold]{config.QDRANT_HOST}:{config.QDRANT_PORT}[/bold]"
            )
        else:
            self.client = QdrantClient(path=config.QDRANT_PATH)
            console.print(f"  [green]✓[/green] Qdrant local: {config.QDRANT_PATH}")

        self._ensure_collection()
        console.print(
            f"  [green]✓[/green] Collection: [bold]{config.QDRANT_COLLECTION}[/bold]\n"
        )

    def _ensure_collection(self) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if config.QDRANT_COLLECTION not in existing:
            self.client.create_collection(
                collection_name=config.QDRANT_COLLECTION,
                vectors_config=VectorParams(
                    size=config.EMBEDDING_DIM,
                    distance=Distance.COSINE,
                ),
            )
            console.print(
                f"  [yellow]Created collection[/yellow] {config.QDRANT_COLLECTION!r}"
            )

    def _embed(self, texts: list[str]) -> list[list[float]]:
        vecs = self.embed_model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False,
        )
        return vecs.tolist()

    def is_empty(self) -> bool:
        return self.client.get_collection(config.QDRANT_COLLECTION).points_count == 0

    def index_documents(self, docs: list[Document], batch_size: int = 32) -> None:
        """Embed and upload documents to Qdrant."""
        console.print("[cyan]⚡ Indexing documents into Qdrant…[/cyan]")
        points: list[PointStruct] = []

        for i in track(range(0, len(docs), batch_size), description="Embedding…"):
            batch   = docs[i : i + batch_size]
            vectors = self._embed([d.text for d in batch])
            for doc, vec in zip(batch, vectors):
                points.append(PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vec,
                    payload={
                        "text":     doc.text,
                        "source":   doc.source,
                        "rel_path": getattr(doc, "rel_path", ""),
                        "page":     doc.page,
                        "chunk_id": doc.chunk_id,
                    },
                ))

        self.client.upsert(collection_name=config.QDRANT_COLLECTION, points=points)
        console.print(f"  [bold green]✓ Indexed {len(points)} vectors[/bold green]\n")

    def search(
        self, query: str, top_k: int | None = None,
    ) -> list[tuple[str, float, dict]]:
        """Semantic search — returns (text, score, metadata) triples."""
        top_k     = top_k or config.TOP_K_QDRANT
        query_vec = self._embed([query])[0]
        
        response = self.client.query_points(
            collection_name=config.QDRANT_COLLECTION,
            query=query_vec,
            limit=top_k,
            with_payload=True,
        )
        hits = response.points
        
        return [
            (
                h.payload.get("text", ""),
                h.score,
                {k: v for k, v in h.payload.items() if k != "text"},
            )
            for h in hits
        ]

    def get_all_texts(self) -> list[str]:
        """Fetch every stored chunk text (for BM25 index building)."""
        texts:  list[str] = []
        offset = None
        while True:
            batch, next_offset = self.client.scroll(
                collection_name=config.QDRANT_COLLECTION,
                limit=256, offset=offset, with_payload=True,
            )
            texts.extend(r.payload.get("text", "") for r in batch)
            if next_offset is None:
                break
            offset = next_offset
        return texts

    def delete_collection(self) -> None:
        """Delete the collection and recreate it (for force re-indexing)."""
        self.client.delete_collection(config.QDRANT_COLLECTION)
        self._ensure_collection()
        console.print(f"  [yellow]Collection '{config.QDRANT_COLLECTION}' reset[/yellow]")
