"""
Unit tests for core/retrieval/intent_router.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from core.retrieval.intent_router import IntentRouter, IntentType
from core.retrieval.contracts import RetrievalChannel

def test_intent_router_temporal_before():
    router = IntentRouter()
    analysis = router.analyze("What did I do before switching to PostgreSQL?")
    assert analysis.intent_type in (IntentType.TEMPORAL, IntentType.HYBRID)
    assert analysis.temporal_direction == "before"
    assert "switching to postgresql" in analysis.temporal_anchor.lower()
    assert analysis.channel_weights[RetrievalChannel.TEMPORAL] >= 0.25
    print("PASS: test_intent_router_temporal_before")

def test_intent_router_entity():
    router = IntentRouter()
    analysis = router.analyze("How is Bob doing on the project?")
    assert analysis.intent_type in (IntentType.ENTITY, IntentType.HYBRID)
    assert any("bob" in e.lower() for e in analysis.entities_detected)
    assert analysis.channel_weights[RetrievalChannel.ENTITY] >= 0.25
    print("PASS: test_intent_router_entity")

def test_intent_router_comparison():
    router = IntentRouter()
    analysis = router.analyze("Compare what I planned for Q1 vs. what was actually finished.")
    assert analysis.intent_type == IntentType.COMPARISON
    assert len(analysis.comparison_targets) >= 2
    assert analysis.channel_weights[RetrievalChannel.TEMPORAL] > 0.1
    print("PASS: test_intent_router_comparison")

def test_intent_router_semantic():
    router = IntentRouter()
    analysis = router.analyze("Tell me about machine learning neural network architectures.")
    assert analysis.intent_type == IntentType.SEMANTIC
    assert analysis.channel_weights[RetrievalChannel.SEMANTIC] >= 0.50
    print("PASS: test_intent_router_semantic")

if __name__ == "__main__":
    test_intent_router_temporal_before()
    test_intent_router_entity()
    test_intent_router_comparison()
    test_intent_router_semantic()
    print("All IntentRouter unit tests passed successfully.")
