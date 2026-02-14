from __future__ import annotations

import hashlib
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from .config import get_settings


class MemoryClient:
    _instance: Optional["MemoryClient"] = None

    def __new__(cls) -> "MemoryClient":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.client = None
        self._collection = None
        self._initialized = True

    def _ensure_connected(self):
        if self.client is not None:
            return

        settings = get_settings()
        host = settings.CHROMA_HOST or "localhost"
        port = settings.CHROMA_PORT or 8000

        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex((host, port))
            sock.close()

            if result == 0:
                self.client = chromadb.HttpClient(
                    host=host, port=port, settings=ChromaSettings(allow_reset=True)
                )
            else:
                self.client = chromadb.PersistentClient(path="./chroma_data")
        except Exception:
            self.client = chromadb.PersistentClient(path="./chroma_data")

    @property
    def collection(self):
        self._ensure_connected()
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name="migration_knowledge",
                metadata={"description": "Semantic mappings and transformation logic"},
            )
        return self._collection

    def store_mapping_logic(
        self, source_col: str, target_col: str, logic: str, metadata: dict = None
    ) -> None:
        mapping_id = hashlib.md5(
            f"{source_col}:{target_col}".encode()
        ).hexdigest()

        meta = metadata or {}
        meta.update({"source": source_col, "target": target_col})

        self.collection.add(
            documents=[logic], metadatas=[meta], ids=[mapping_id]
        )

    def find_similar_transformation(
        self, source_col_desc: str, target_col_desc: str, top_k: int = 1
    ) -> Optional[str]:
        query_text = f"{source_col_desc} -> {target_col_desc}"

        results = self.collection.query(
            query_texts=[query_text], n_results=top_k
        )

        if results and results["documents"] and results["documents"][0]:
            return results["documents"][0][0]

        return None

    def reset(self) -> None:
        self._ensure_connected()
        try:
            self.client.delete_collection(name="migration_knowledge")
        except Exception:
            pass
        self._collection = None
