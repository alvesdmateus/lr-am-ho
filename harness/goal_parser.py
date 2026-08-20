import uuid
from harness.state import Goal


class GoalParser:
    """
    Harness-owned: Normalizes raw natural language input into a structured Goal.
    No LLM needed — this is pure schema enforcement and normalization.
    """

    def parse(self, raw_input: str) -> Goal:
        """
        Converts raw text input into a Goal object.
        If the input contains structured hints (bullet points, numbered lists),
        they are extracted as preliminary definition_of_done items.
        """
        objective = raw_input.strip()
        if not objective:
            raise ValueError("Goal objective cannot be empty.")

        # Extract preliminary DoD hints from bullet points or numbered items
        dod_hints = self._extract_dod_hints(objective)

        return Goal(
            id=f"goal_{uuid.uuid4().hex[:8]}",
            objective=objective,
            definition_of_done=dod_hints,
        )

    def _extract_dod_hints(self, text: str) -> list:
        """
        Extracts structured hints from the input text.
        Looks for lines starting with -, *, or numbered patterns (1., 2., etc.).
        These are preliminary hints, NOT the final DoD — the Manager LLM refines them.
        """
        hints = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith(("- ", "* ", "• ")):
                hints.append(stripped.lstrip("-*• ").strip())
            elif len(stripped) > 2 and stripped[0].isdigit() and stripped[1] in (".", ")"):
                hints.append(stripped[2:].strip())
        return hints
