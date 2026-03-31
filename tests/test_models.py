"""Tests for SDK models."""

import pytest
from wauldo.models import (
    ReasoningResult,
    ConceptResult,
    PlanResult,
    PlanStep,
    Concept,
)


class TestReasoningResult:
    def test_from_content_basic(self):
        content = """# Tree-of-Thought Analysis

## Problem
Test problem

## Thought Tree
Branch 1: Consider option A
Branch 2: Consider option B

## Solution:
Option A is better because of efficiency.
"""
        result = ReasoningResult.from_content(content, "Test problem", 3, 3)

        assert result.problem == "Test problem"
        assert result.depth == 3
        assert result.branches == 3
        assert "efficiency" in result.solution.lower()

    def test_from_content_empty(self):
        result = ReasoningResult.from_content("", "Empty", 1, 1)
        assert result.problem == "Empty"
        assert result.raw_content == ""


class TestConceptResult:
    def test_from_content_with_concepts(self):
        content = """Extracted concepts:
- Authentication
- JWT Token
- User Session
"""
        result = ConceptResult.from_content(content, "code")

        assert result.source_type == "code"
        assert len(result.concepts) == 3
        assert result.concepts[0].name == "Authentication"

    def test_from_content_empty(self):
        result = ConceptResult.from_content("No concepts", "text")
        assert result.source_type == "text"
        assert len(result.concepts) == 0


class TestPlanResult:
    def test_from_content_with_steps(self):
        content = """# Task Plan: Test task

**Category**: Implementation
**Steps**: 3

---

1. First step
2. Second step
3. Third step

---

## Summary

- **Total steps**: 3
- **Estimated total effort**: 2 hours 30 min
"""
        result = PlanResult.from_content(content, "Test task")

        assert result.task == "Test task"
        assert result.category == "Implementation"
        assert len(result.steps) == 3
        assert result.steps[0].title == "First step"
        assert result.total_effort == "2 hours 30 min"

    def test_from_content_empty(self):
        result = PlanResult.from_content("Empty plan", "Empty task")
        assert result.task == "Empty task"
        assert len(result.steps) == 0


class TestPlanStep:
    def test_creation(self):
        step = PlanStep(
            number=1,
            title="Test step",
            description="Test description",
            priority="High",
            effort="1 hour",
            dependencies=["Step 0"],
        )

        assert step.number == 1
        assert step.title == "Test step"
        assert step.priority == "High"
        assert len(step.dependencies) == 1


class TestConcept:
    def test_creation(self):
        concept = Concept(
            name="Test",
            concept_type="Entity",
            weight=0.8,
            description="Test concept",
        )

        assert concept.name == "Test"
        assert concept.weight == 0.8

    def test_weight_bounds(self):
        # Valid weights
        Concept(name="Low", concept_type="Entity", weight=0.0)
        Concept(name="High", concept_type="Entity", weight=1.0)

        # Invalid weights should raise validation error
        with pytest.raises(Exception):  # Pydantic validation error
            Concept(name="Invalid", concept_type="Entity", weight=1.5)
