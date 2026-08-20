import os
import json
import time
from typing import Dict, Any, Optional, List
from pydantic import BaseModel


class StateStore:
    """Handles persistence of Goal, Task Graph, Event Logs, and Checkpoints."""
    def __init__(self, storage_dir: str = ".harness/state"):
        self.storage_dir = storage_dir
        self.state_file = os.path.join(storage_dir, "harness_state.json")
        self.events_file = os.path.join(storage_dir, "events.jsonl")
        self.checkpoints_dir = os.path.join(storage_dir, "checkpoints")
        
        os.makedirs(self.storage_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)

    def save_state(self, state_dict: Dict[str, Any]) -> None:
        """Saves current harness state snapshot."""
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(state_dict, f, indent=2)

    def load_state(self) -> Optional[Dict[str, Any]]:
        """Loads harness state snapshot if it exists."""
        if not os.path.exists(self.state_file):
            return None
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def log_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Appends an event to the append-only events log."""
        record = {
            "timestamp": time.time(),
            "event_type": event_type,
            "payload": payload
        }
        with open(self.events_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def create_checkpoint(self, checkpoint_id: str, state_dict: Dict[str, Any]) -> str:
        """Saves an immutable point-in-time checkpoint."""
        filepath = os.path.join(self.checkpoints_dir, f"checkpoint_{checkpoint_id}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(state_dict, f, indent=2)
        return filepath
