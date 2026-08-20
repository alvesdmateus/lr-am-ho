import os
import shutil
import unittest
from storage.memory_store import MemoryTier, MemoryCategory, MemoryEntry
from agent.memory import TieredMemoryManager
from harness.state import Task, ExecutionResult, EvaluationVerdict


class TestTieredMemoryManager(unittest.TestCase):
    def setUp(self):
        self.test_dir = ".test_harness_mem"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)
        self.memory = TieredMemoryManager(
            hot_threshold=0.3,
            cold_threshold=0.3,
            promotion_pass_threshold=3,
            storage_dir=f"{self.test_dir}/storage",
        )

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_cache_hit_and_miss(self):
        # 1. Miss on empty cache
        lookup = self.memory.retrieve("build postgres docker container")
        self.assertFalse(lookup.is_hit)
        self.assertEqual(lookup.cache_stats["misses"], 1)

        # 2. Add Hot Observation
        self.memory.record_observation(
            content="Use port 5433 to avoid postgres collisions",
            category=MemoryCategory.ENVIRONMENT,
        )

        # 3. Lookup should hit Hot Cache
        lookup2 = self.memory.retrieve("postgres port collisions")
        self.assertTrue(lookup2.is_hit)
        self.assertEqual(lookup2.source_tier, MemoryTier.HOT)
        self.assertEqual(lookup2.cache_stats["hot_hits"], 1)
        self.assertIn("5433", lookup2.entries[0].content)

    def test_promotion_pipeline_hot_to_cold(self):
        # Add a tentative hot memory item
        entry = self.memory.record_observation(
            content="Always run tests with pytest -v",
            category=MemoryCategory.CONVENTION,
        )
        self.assertEqual(entry.tier, MemoryTier.HOT)
        self.assertEqual(entry.base_weight, 0.5)

        # Pass 1
        self.memory.handle_evaluator_feedback(validated_ids=[entry.id], invalidated_ids=[], passed=True)
        self.assertEqual(self.memory.hot_cache[entry.id].validation_passes, 1)
        self.assertEqual(self.memory.hot_cache[entry.id].tier, MemoryTier.HOT)

        # Pass 2
        self.memory.handle_evaluator_feedback(validated_ids=[entry.id], invalidated_ids=[], passed=True)
        self.assertEqual(self.memory.hot_cache[entry.id].validation_passes, 2)
        self.assertEqual(self.memory.hot_cache[entry.id].tier, MemoryTier.HOT)

        # Pass 3 -> PROMOTION TO COLD
        initial_hot_id = entry.id
        self.memory.handle_evaluator_feedback(validated_ids=[initial_hot_id], invalidated_ids=[], passed=True)
        self.assertNotIn(initial_hot_id, self.memory.hot_cache)
        
        # Cold store entry should exist with higher weight
        expected_cold_id = f"cold_{initial_hot_id.replace('hot_', '')}"
        self.assertIn(expected_cold_id, self.memory.cold_store)
        promoted = self.memory.cold_store[expected_cold_id]
        self.assertEqual(promoted.tier, MemoryTier.COLD)
        self.assertEqual(promoted.base_weight, 0.9)

    def test_failure_penalty_and_invalidation(self):
        entry = self.memory.record_observation(
            content="Use port 3000 for web mock server",
            category=MemoryCategory.ENVIRONMENT,
        )
        initial_confidence = entry.confidence

        # Evaluation fails using this memory
        self.memory.handle_evaluator_feedback(validated_ids=[], invalidated_ids=[entry.id], passed=False)
        self.assertLess(entry.confidence, initial_confidence)

        # Multiple failures cause superseded = True
        self.memory.handle_evaluator_feedback(validated_ids=[], invalidated_ids=[entry.id], passed=False)
        self.memory.handle_evaluator_feedback(validated_ids=[], invalidated_ids=[entry.id], passed=False)
        self.assertTrue(entry.superseded)

        # Garbage collection should evict superseded items
        stats = self.memory.garbage_collect()
        self.assertEqual(stats["purged_hot"], 1)
        self.assertNotIn(entry.id, self.memory.hot_cache)

    def test_epistemic_prompt_briefing_formatting(self):
        # Add cold item
        cold_entry = MemoryEntry(
            id="cold_101",
            content="Strictly type check with mypy --strict",
            category=MemoryCategory.CONVENTION,
            tier=MemoryTier.COLD,
            base_weight=0.9,
            confidence=1.0,
        )
        self.memory.cold_store[cold_entry.id] = cold_entry

        # Add hot item with typing relevance
        hot_entry = self.memory.record_observation(
            content="Mypy typing checks require pydantic plugin",
            category=MemoryCategory.ENVIRONMENT,
        )

        task = Task(
            id="t1",
            title="Setup mypy typing checks",
            description="Ensure mypy passes across codebase",
            definition_of_done=["mypy type check passes"],
        )

        briefing = self.memory.format_prompt_briefing(task)
        self.assertIn("=== AGENT MEMORY BRIEFING ===", briefing)
        self.assertIn("SYSTEM CONVENTIONS", briefing)
        self.assertIn("RECENT OBSERVATIONS", briefing)
        self.assertIn("mypy", briefing)
        self.assertIn(cold_entry.id, task.used_memory_ids)


if __name__ == "__main__":
    unittest.main()
