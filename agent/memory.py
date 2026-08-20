import time
import math
from typing import List, Optional, Tuple, Dict, Any
from storage.memory_store import (
    MemoryEntry,
    MemoryTier,
    MemoryCategory,
    CacheLookupResult,
    PersistentMemoryStore,
)
from harness.state import Task, ExecutionResult, EvaluationVerdict


class TieredMemoryManager:
    """
    Active Memory Agent implementing the 7 core memory management principles:
    1. Epistemic Hygiene (Stateless workers, stateful harness)
    2. Distillation Over Ingestion (Episodic -> Semantic)
    3. Grounding in Git / FS (Meta-knowledge over code duplication)
    4. Precision Over Recall (Strict token caps & threshold filters)
    5. Bi-Temporal Invalidation & Decay
    6. Evaluator-Driven Promotion (Hot -> Cold after N passes)
    7. Structured Epistemic Tagging
    """

    def __init__(
        self,
        hot_threshold: float = 0.50,
        cold_threshold: float = 0.45,
        promotion_pass_threshold: int = 3,
        max_briefing_items: int = 5,
        max_briefing_tokens: int = 500,
        storage_dir: str = ".harness/storage",
    ):
        self.hot_threshold = hot_threshold
        self.cold_threshold = cold_threshold
        self.promotion_pass_threshold = promotion_pass_threshold
        self.max_briefing_items = max_briefing_items
        self.max_briefing_tokens = max_briefing_tokens

        self.persistent_store = PersistentMemoryStore(storage_dir)

        # In-Memory Hot Cache (Volatile, tentative, weight ~0.5)
        self.hot_cache: Dict[str, MemoryEntry] = {}

        # Cold Store (Persistent, verified, weight ~0.9)
        self.cold_store: Dict[str, MemoryEntry] = self.persistent_store.load_cold_entries()

        # Cache metrics
        self.stats = {"hot_hits": 0, "cold_hits": 0, "misses": 0}

    # -------------------------------------------------------------------------
    # Similarity & Scoring (Principle 4: Precision over Recall)
    # -------------------------------------------------------------------------
    def _compute_similarity(self, query: str, text: str) -> float:
        """Computes semantic/keyword coverage between query and memory item."""
        if not query or not text:
            return 0.0
        # Clean punctuation and normalize words
        clean_q = "".join([c if c.isalnum() or c.isspace() else " " for c in query.lower()])
        clean_t = "".join([c if c.isalnum() or c.isspace() else " " for c in text.lower()])
        words_q = set(clean_q.split())
        words_t = set(clean_t.split())
        if not words_q or not words_t:
            return 0.0
        intersection = words_q.intersection(words_t)
        if not intersection:
            return 0.0
        # Coverage of memory entry terms in the query
        entry_coverage = len(intersection) / len(words_t)
        query_coverage = len(intersection) / len(words_q)
        return round((0.75 * entry_coverage) + (0.25 * query_coverage), 4)

    def _compute_decay_factor(self, entry: MemoryEntry) -> float:
        """Principle 5: Time decay for unpromoted hot memories; slow decay for cold."""
        age_hours = (time.time() - entry.created_at) / 3600.0
        if entry.tier == MemoryTier.HOT:
            # Hot memories have a 6-hour half-life if not accessed/promoted
            return math.exp(-0.115 * age_hours)
        else:
            # Cold memories have a 30-day half-life
            return math.exp(-0.001 * age_hours)

    # -------------------------------------------------------------------------
    # Retrieval (Principle 4 & 7)
    # -------------------------------------------------------------------------
    def retrieve(self, query: str) -> CacheLookupResult:
        """
        Two-tier lookup:
        1. Checks Hot Cache
        2. Checks Cold Store
        3. Returns ranked results with hit/miss classification
        """
        candidates: List[Tuple[float, MemoryEntry]] = []
        hot_hit = False
        cold_hit = False

        # 1. Check Hot Cache
        for entry in list(self.hot_cache.values()):
            if entry.superseded:
                continue
            sim = self._compute_similarity(query, entry.content)
            if sim >= self.hot_threshold:
                hot_hit = True
                entry.hit_count += 1
                entry.last_hit_at = time.time()
                decay = self._compute_decay_factor(entry)
                score = sim * entry.base_weight * entry.confidence * decay
                candidates.append((score, entry))

        # 2. Check Cold Store
        for entry in list(self.cold_store.values()):
            if entry.superseded:
                continue
            sim = self._compute_similarity(query, entry.content)
            if sim >= self.cold_threshold:
                cold_hit = True
                entry.hit_count += 1
                entry.last_hit_at = time.time()
                decay = self._compute_decay_factor(entry)
                score = sim * entry.base_weight * entry.confidence * decay
                candidates.append((score, entry))

        # Record Hit/Miss Metrics
        if hot_hit:
            self.stats["hot_hits"] += 1
        elif cold_hit:
            self.stats["cold_hits"] += 1
        else:
            self.stats["misses"] += 1

        # Sort by final score
        candidates.sort(key=lambda x: x[0], reverse=True)
        results = [c[1] for c in candidates[: self.max_briefing_items]]

        tier_found = (
            MemoryTier.HOT if hot_hit else (MemoryTier.COLD if cold_hit else None)
        )

        return CacheLookupResult(
            is_hit=(hot_hit or cold_hit),
            source_tier=tier_found,
            entries=results,
            cache_stats=self.stats.copy(),
            retrieval_query=query,
        )

    # -------------------------------------------------------------------------
    # Epistemic Prompt Formatting (Principle 1 & 7)
    # -------------------------------------------------------------------------
    def format_prompt_briefing(self, task: Task) -> str:
        """
        Constructs a structured, token-capped briefing for the Worker.
        Categorizes by authority level: [VERIFIED RULES] vs [RECENT OBSERVATIONS].
        """
        query = f"{task.title} {task.description} {' '.join(task.definition_of_done)}"
        lookup = self.retrieve(query)

        if not lookup.is_hit or not lookup.entries:
            return "No previous memory or specific conventions found for this task."

        cold_items = [e for e in lookup.entries if e.tier == MemoryTier.COLD]
        hot_items = [e for e in lookup.entries if e.tier == MemoryTier.HOT]

        lines = ["=== AGENT MEMORY BRIEFING ==="]

        if cold_items:
            lines.append("### 🛡️ SYSTEM CONVENTIONS (Verified - Must Follow)")
            for item in cold_items:
                lines.append(f"- [{item.category.value.upper()}] {item.content} (id: {item.id})")

        if hot_items:
            lines.append("\n### 💡 RECENT OBSERVATIONS (Tentative - Contextual Hints)")
            for item in hot_items:
                lines.append(f"- [{item.category.value.upper()}] {item.content} (id: {item.id})")

        lines.append("=============================")

        # Update task with used memory ids
        task.used_memory_ids = [e.id for e in lookup.entries]

        briefing_text = "\n".join(lines)
        # Approximate token budget clamp
        words = briefing_text.split()
        if len(words) > self.max_briefing_tokens:
            briefing_text = " ".join(words[: self.max_briefing_tokens]) + "\n...[truncated]"

        return briefing_text

    # -------------------------------------------------------------------------
    # Distillation & Consolidation (Principle 2 & 3)
    # -------------------------------------------------------------------------
    def record_observation(
        self,
        content: str,
        category: MemoryCategory = MemoryCategory.CONVENTION,
        source_task_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> MemoryEntry:
        """Adds a tentative observation into HOT memory."""
        # Principle 3: Filter out raw code dumps
        if "```" in content or len(content.splitlines()) > 5:
            # Compress / clean up code dumps to keep meta-knowledge only
            content = content.replace("```", "").strip()
            content = " ".join(content.split())[:200]

        item_id = f"hot_{int(time.time() * 1000)}"
        entry = MemoryEntry(
            id=item_id,
            content=content,
            category=category,
            tier=MemoryTier.HOT,
            base_weight=0.5,
            confidence=0.7,
            source_task_id=source_task_id,
            tags=tags or [],
        )
        self.hot_cache[item_id] = entry
        return entry

    def consolidate_episode(
        self,
        task: Task,
        execution_result: ExecutionResult,
        verdict: EvaluationVerdict,
    ) -> List[MemoryEntry]:
        """
        Principle 2: Distills raw logs and verdict into actionable structured memories.
        """
        new_entries: List[MemoryEntry] = []

        # 1. Distill discoveries found by worker
        for learning in execution_result.discovered_learnings:
            entry = self.record_observation(
                content=learning,
                category=MemoryCategory.CONVENTION,
                source_task_id=task.id,
                tags=[task.id, "discovery"],
            )
            new_entries.append(entry)

        # 2. Distill failure traps if task failed
        if not verdict.passed and verdict.feedback:
            trap_content = f"Failure Trap in {task.title}: {verdict.feedback}"
            entry = self.record_observation(
                content=trap_content,
                category=MemoryCategory.FAILURE_TRAP,
                source_task_id=task.id,
                tags=[task.id, "failure_trap"],
            )
            new_entries.append(entry)

        # 3. Handle promotion & feedback for memories that were active
        self.handle_evaluator_feedback(
            validated_ids=verdict.validated_memory_ids or task.used_memory_ids if verdict.passed else [],
            invalidated_ids=verdict.invalidated_memory_ids or task.used_memory_ids if not verdict.passed else [],
            passed=verdict.passed,
        )

        return new_entries

    # -------------------------------------------------------------------------
    # Evaluator-Driven Promotion & Invalidation (Principle 5 & 6)
    # -------------------------------------------------------------------------
    def handle_evaluator_feedback(
        self,
        validated_ids: List[str],
        invalidated_ids: List[str],
        passed: bool,
    ) -> None:
        """
        Principle 6: Evaluator reinforces or penalizes memories.
        - Validated items: +0.1 confidence, increments validation passes.
          If passes >= promotion threshold -> Promoted to COLD.
        - Invalidated items: -0.3 confidence. If confidence < 0.2 -> superseded.
        """
        # Handle validated memories
        for mem_id in validated_ids:
            entry = self.hot_cache.get(mem_id) or self.cold_store.get(mem_id)
            if not entry:
                continue

            entry.validation_passes += 1
            entry.confidence = min(1.0, round(entry.confidence + 0.1, 2))

            # Promotion gate: HOT -> COLD
            if (
                entry.tier == MemoryTier.HOT
                and entry.validation_passes >= self.promotion_pass_threshold
            ):
                self._promote_to_cold(entry)

        # Handle invalidated memories
        for mem_id in invalidated_ids:
            entry = self.hot_cache.get(mem_id) or self.cold_store.get(mem_id)
            if not entry:
                continue

            entry.confidence = max(0.0, round(entry.confidence - 0.3, 2))
            if entry.confidence < 0.2:
                entry.superseded = True

        # Persist cold updates
        self._sync_cold_store()

    def _promote_to_cold(self, entry: MemoryEntry) -> None:
        """Promotes a validated hot entry to persistent cold memory."""
        old_id = entry.id
        self.hot_cache.pop(old_id, None)
        entry.tier = MemoryTier.COLD
        if not entry.id.startswith("cold_"):
            entry.id = f"cold_{entry.id.replace('hot_', '')}"
        entry.base_weight = 0.9  # Upgraded trust weight
        self.cold_store[entry.id] = entry

    def _sync_cold_store(self) -> None:
        """Saves current cold memory to disk."""
        self.persistent_store.save_cold_entries(self.cold_store)

    # -------------------------------------------------------------------------
    # Maintenance / Garbage Collection (Principle 5)
    # -------------------------------------------------------------------------
    def garbage_collect(self) -> Dict[str, int]:
        """Evicts superseded or dead memories."""
        purged_hot = 0
        purged_cold = 0

        # Purge dead hot cache items
        for k in list(self.hot_cache.keys()):
            if self.hot_cache[k].superseded or self.hot_cache[k].confidence <= 0.0:
                del self.hot_cache[k]
                purged_hot += 1

        # Purge dead cold store items
        for k in list(self.cold_store.keys()):
            if self.cold_store[k].superseded or self.cold_store[k].confidence <= 0.0:
                del self.cold_store[k]
                purged_cold += 1

        if purged_cold > 0:
            self._sync_cold_store()

        return {"purged_hot": purged_hot, "purged_cold": purged_cold}
