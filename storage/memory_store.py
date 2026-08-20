import os
import json
import time
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field


class MemoryTier(str, Enum):
    HOT = "hot"     # Fast, in-memory, volatile, tentative (Weight ~0.5)
    COLD = "cold"   # Persistent, verified, battle-tested (Weight ~0.9)


class MemoryCategory(str, Enum):
    CONVENTION = "convention"         # Code style, formatting, repo patterns
    FAILURE_TRAP = "failure_trap"     # Known traps, debugging fixes, errors to avoid
    ARCH_DECISION = "arch_decision"   # Structural choices, interfaces, DB models
    ENVIRONMENT = "environment"       # Port bindings, env vars, build flags


class MemoryEntry(BaseModel):
    id: str
    content: str
    category: MemoryCategory = MemoryCategory.CONVENTION
    tier: MemoryTier = MemoryTier.HOT
    base_weight: float = 0.5          # Hot: 0.5, Cold: 0.9
    confidence: float = 1.0           # 0.0 to 1.0
    access_count: int = 0
    hit_count: int = 0
    created_at: float = Field(default_factory=time.time)
    last_hit_at: Optional[float] = None
    validation_passes: int = 0        # Passes before Hot -> Cold promotion
    source_task_id: Optional[str] = None
    superseded: bool = False
    superseded_by: Optional[str] = None
    tags: List[str] = Field(default_factory=list)

    @property
    def effective_score(self) -> float:
        """Computes current authority score based on tier weight, confidence and validity."""
        if self.superseded:
            return 0.0
        return round(self.base_weight * self.confidence, 4)


class CacheLookupResult(BaseModel):
    is_hit: bool
    source_tier: Optional[MemoryTier] = None
    entries: List[MemoryEntry] = Field(default_factory=list)
    cache_stats: Dict[str, int] = Field(default_factory=dict)
    retrieval_query: str = ""


class PersistentMemoryStore:
    """Disk-backed storage for Cold, consolidated long-term memory."""
    def __init__(self, storage_dir: str = ".harness/storage"):
        self.storage_dir = storage_dir
        self.file_path = os.path.join(storage_dir, "cold_memory.json")
        os.makedirs(self.storage_dir, exist_ok=True)

    def save_cold_entries(self, entries: Dict[str, MemoryEntry]) -> None:
        data = {k: v.model_dump() for k, v in entries.items()}
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_cold_entries(self) -> Dict[str, MemoryEntry]:
        if not os.path.exists(self.file_path):
            return {}
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {k: MemoryEntry(**v) for k, v in data.items()}
        except Exception:
            return {}
