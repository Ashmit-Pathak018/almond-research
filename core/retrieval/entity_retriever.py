"""
Project Almond V3 — Entity Retriever
Retrieves candidates through canonical entity and alias resolution via EntityStore.
"""

from __future__ import annotations
import logging
import re
from typing import List, Optional, Set, Dict, Any

from core.knowledge.entity_store import EntityStore, AliasResolution
from core.retrieval.contracts import RetrievalCandidate, RetrievalChannel, RetrievalQuery

logger = logging.getLogger(__name__)


class EntityRetriever:
    """
    Retrieves memories linked to entities mentioned in the query.
    Utilizes EntityStore for deterministic alias resolution (e.g. 'Bob' -> 'Robert Vance').
    """

    def __init__(self, entity_store: EntityStore):
        self.entity_store = entity_store

    def retrieve(
        self,
        query: RetrievalQuery,
        entity_hints: Optional[List[str]] = None,
        top_k: Optional[int] = None
    ) -> List[RetrievalCandidate]:
        k = top_k or query.top_k
        namespace_id = query.namespace_id

        # 1. Identify candidate entity phrases from hints or query tokens
        candidate_mentions: Set[str] = set()
        if entity_hints:
            candidate_mentions.update(entity_hints)

        # Extract capitalized sequences and words
        raw_words = query.query_text.split()
        for idx, w in enumerate(raw_words):
            clean = re.sub(r"[^\w]", "", w)
            if clean and clean[0].isupper() and clean.lower() not in ["what", "how", "when", "why", "where", "who", "did", "was", "the", "i", "compare"]:
                candidate_mentions.add(clean)

        # Also extract multi-word capitalized phrases (e.g., "Robert Vance")
        phrase_matches = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", query.query_text)
        for pm in phrase_matches:
            candidate_mentions.add(pm)

        if not candidate_mentions:
            return []

        # 2. Resolve aliases via EntityStore
        resolutions: List[AliasResolution] = []
        for mention in candidate_mentions:
            res = self.entity_store.resolve_alias(mention, namespace_id)
            if res:
                resolutions.append(res)

        if not resolutions:
            return []

        # 3. Retrieve memories linked to resolved entities
        candidates_map: Dict[str, RetrievalCandidate] = {}

        for res in resolutions:
            eid = res.entity.entity_id
            cname = res.entity.canonical_name
            conf = res.confidence

            memories_linked = self.entity_store.get_memories_for_entity(eid)
            for link in memories_linked:
                mid = link["memory_id"] if isinstance(link, dict) else getattr(link, "memory_id")
                link_base_conf = link["confidence"] if isinstance(link, dict) else getattr(link, "confidence", 1.0)
                link_conf = float(link_base_conf) * conf

                if mid not in candidates_map:
                    cand = RetrievalCandidate(
                        memory_id=mid,
                        entity_matches=[cname],
                        entity_confidence=link_conf
                    )
                    candidates_map[mid] = cand
                else:
                    cand = candidates_map[mid]
                    if cname not in cand.entity_matches:
                        cand.entity_matches.append(cname)
                    cand.entity_confidence = max(cand.entity_confidence, link_conf)

                cand.add_channel_result(
                    channel=RetrievalChannel.ENTITY,
                    raw_score=conf,
                    normalized_score=link_conf,
                    metadata={
                        "matched_alias": res.matched_alias,
                        "canonical_name": cname,
                        "confidence": link_conf,
                        "resolution_type": res.resolution_type
                    }
                )

        candidate_list = list(candidates_map.values())
        candidate_list.sort(key=lambda c: c.entity_confidence, reverse=True)
        return candidate_list[:k]
