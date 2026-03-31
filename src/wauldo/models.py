"""
Pydantic models for Wauldo SDK.
"""

import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# Compiled once at module level (not in hot-path methods)
_STEP_PATTERN = re.compile(r"^(\d+)\.\s+(.+)$")


# Base response model
class ToolResponse(BaseModel):
    """Base model for tool responses."""

    content: str
    is_error: bool = False


# Reasoning models
class ReasoningResult(BaseModel):
    """Result from Tree-of-Thought reasoning."""

    problem: str
    solution: str
    thought_tree: str
    depth: int
    branches: int
    raw_content: str = ""

    @classmethod
    def from_content(cls, content: str, problem: str, depth: int, branches: int) -> "ReasoningResult":
        """Parse reasoning result from raw content (JSON or markdown fallback)."""
        # Try JSON first (structured output from server v0.2+)
        try:
            import json
            data = json.loads(content)
            if "solution" in data:
                return cls(
                    problem=data.get("problem", problem),
                    solution=data["solution"],
                    thought_tree=data.get("thought_tree", content),
                    depth=data.get("depth", depth),
                    branches=data.get("branches", branches),
                    raw_content=content,
                )
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

        # Fallback: markdown heuristic parser
        lines = content.split("\n")
        solution = ""
        in_solution = False

        for line in lines:
            if "Solution:" in line or "Best path:" in line:
                in_solution = True
                continue
            if in_solution and line.strip():
                solution = line.strip()
                break

        return cls(
            problem=problem,
            solution=solution or "See thought tree for analysis",
            thought_tree=content,
            depth=depth,
            branches=branches,
            raw_content=content,
        )


# Concept models
class Concept(BaseModel):
    """A single extracted concept."""

    name: str
    concept_type: str
    weight: float = Field(ge=0.0, le=1.0)
    description: Optional[str] = None


class ConceptResult(BaseModel):
    """Result from concept extraction."""

    concepts: List[Concept]
    source_type: str
    raw_content: str = ""

    @classmethod
    def from_content(cls, content: str, source_type: str) -> "ConceptResult":
        """Parse concept result from raw content (JSON or markdown fallback)."""
        # Try JSON first (structured output from server v0.2+)
        try:
            import json
            data = json.loads(content)
            if "concepts" in data:
                return cls(
                    concepts=[
                        Concept(
                            name=c["name"],
                            concept_type=c.get("concept_type", "Entity"),
                            weight=c.get("weight", 0.8),
                            description=c.get("description"),
                        )
                        for c in data["concepts"]
                    ],
                    source_type=data.get("source_type", source_type),
                    raw_content=content,
                )
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

        # Fallback: markdown heuristic parser
        concepts: List[Concept] = []

        lines = content.split("\n")
        for line in lines:
            if line.strip().startswith("- "):
                name = line.strip()[2:].split(":")[0].strip()
                if name:
                    concepts.append(
                        Concept(
                            name=name,
                            concept_type="Entity",
                            weight=0.8,
                        )
                    )

        return cls(
            concepts=concepts,
            source_type=source_type,
            raw_content=content,
        )


# Long context models
class Chunk(BaseModel):
    """A text chunk from document processing."""

    id: str
    content: str
    position: int
    priority: str = "Medium"


class ChunkResult(BaseModel):
    """Result from document chunking."""

    chunks: List[Chunk]
    total_chunks: int
    raw_content: str = ""


class RetrievalResult(BaseModel):
    """Result from context retrieval."""

    query: str
    results: List[Chunk]
    raw_content: str = ""


# Knowledge graph models
class GraphNode(BaseModel):
    """A node in the knowledge graph."""

    id: str
    name: str
    node_type: str
    weight: float = 1.0


class KnowledgeGraphResult(BaseModel):
    """Result from knowledge graph operations."""

    operation: str
    nodes: List[GraphNode] = []
    stats: Optional[Dict[str, Any]] = None
    raw_content: str = ""


# Planning models
class PlanStep(BaseModel):
    """A single step in a task plan."""

    number: int
    title: str
    description: str = ""
    priority: str = "Medium"
    effort: str = ""
    dependencies: List[str] = []


class PlanResult(BaseModel):
    """Result from task planning."""

    task: str
    category: str
    steps: List[PlanStep]
    total_effort: str = ""
    raw_content: str = ""

    @classmethod
    def from_content(cls, content: str, task: str) -> "PlanResult":
        """Parse plan result from raw content (JSON or markdown fallback)."""
        # Try JSON first (structured output from server v0.2+)
        try:
            import json
            data = json.loads(content)
            if "steps" in data:
                return cls(
                    task=data.get("task", task),
                    category=data.get("category", "General"),
                    steps=[
                        PlanStep(
                            number=s.get("number", i + 1),
                            title=s["title"],
                            description=s.get("description", ""),
                            priority=s.get("priority", "Medium"),
                            effort=s.get("effort", ""),
                            dependencies=s.get("dependencies", []),
                        )
                        for i, s in enumerate(data["steps"])
                    ],
                    total_effort=data.get("total_effort", ""),
                    raw_content=content,
                )
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

        # Fallback: markdown heuristic parser
        steps: List[PlanStep] = []
        category = "General"
        total_effort = ""

        lines = content.split("\n")
        current_step = 0

        step_pattern = _STEP_PATTERN

        for line in lines:
            trimmed = line.strip()

            # Extract category: "**Category**: X"
            if trimmed.startswith("**Category**:"):
                category = trimmed.split(":", 1)[1].strip() or "General"
                continue

            # Match numbered steps strictly: "1. Title" (digits + dot + space + text)
            match = step_pattern.match(trimmed)
            if match:
                current_step += 1
                title = match.group(2).strip()
                if title:
                    steps.append(
                        PlanStep(
                            number=current_step,
                            title=title,
                        )
                    )
                continue

            # Extract total effort (may be prefixed with "- ")
            effort_marker = "**Estimated total effort**:"
            idx = trimmed.find(effort_marker)
            if idx >= 0:
                total_effort = trimmed[idx + len(effort_marker):].strip()

        return cls(
            task=task,
            category=category,
            steps=steps,
            total_effort=total_effort,
            raw_content=content,
        )


# Tool definitions
class ToolDefinition(BaseModel):
    """MCP tool definition."""

    name: str
    description: str
    input_schema: Optional[Dict[str, Any]] = None


class ToolsListResult(BaseModel):
    """Result from tools/list."""

    tools: List[ToolDefinition]
