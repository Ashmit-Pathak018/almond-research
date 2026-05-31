"""
entity_extractor.py
-------------------
Phase 2 — Named entity extraction and registry.

Two jobs:
  1. Extract entity mentions from memory text
     ("my phone", "Samsung Galaxy S22", "Ashmit", "the Almond project")

  2. Link mentions to the entity registry
     - Exact match:    "Samsung Galaxy S22" → existing entity
     - Fuzzy match:    "Galaxy S22" → same entity, new alias added
     - Shorthand:      "my phone" → resolved via entity type + recency
     - New entity:     no match → created, flagged if possible duplicate exists

Design decisions
----------------
1. Alias learning on every match.
   Every surface form that resolves to an entity is recorded as an alias.
   This means resolution improves over time rather than drifting.

2. Shorthand resolution via entity type + recency.
   "my phone" → most recently mentioned DEVICE entity.
   "my laptop" → most recently mentioned DEVICE entity.
   This handles the most common coreference patterns without an LLM.

3. needs_review flag instead of blocking.
   When similarity is in the 0.6–0.85 range we don't block or guess —
   we create the entity and flag it so the consolidator can merge later.

4. The EntityRegistry is an in-memory store for Phase 2.
   It will be replaced by a SQLite-backed store in Phase 3 when the
   timeline index is introduced. The interface stays identical.

5. Single LLM call per memory for extraction.
   The LLM identifies raw mentions; resolution is pure Python.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class EntityType(str, Enum):
    PERSON       = "person"
    DEVICE       = "device"
    PLACE        = "place"
    PROJECT      = "project"
    ORGANIZATION = "organization"
    CONCEPT      = "concept"
    UNKNOWN      = "unknown"


@dataclass
class Entity:
    id:           str
    name:         str                        # canonical / best name
    type:         EntityType
    aliases:      set[str] = field(default_factory=set)
    first_seen:   Optional[datetime] = None
    last_seen:    Optional[datetime] = None
    memory_ids:   list[str] = field(default_factory=list)
    fact_ids:     list[str] = field(default_factory=list)
    needs_review: bool = False               # possible duplicate flag
    # salience grows as more memories reference this entity
    reference_count: int = 0

    def all_names(self) -> set[str]:
        """All known surface forms — canonical name plus all aliases."""
        return {self.name} | self.aliases

    def to_dict(self) -> dict:
        return {
            "id":            self.id,
            "name":          self.name,
            "type":          self.type.value,
            "aliases":       sorted(self.aliases),
            "first_seen":    self.first_seen.isoformat() if self.first_seen else None,
            "last_seen":     self.last_seen.isoformat()  if self.last_seen  else None,
            "memory_ids":    self.memory_ids,
            "needs_review":  self.needs_review,
            "reference_count": self.reference_count,
        }


@dataclass
class EntityMention:
    """A raw mention before it has been resolved to an Entity."""
    surface_form: str        # exactly as it appeared ("my phone", "Galaxy S22")
    entity_type:  EntityType
    is_shorthand: bool       # True for "my phone", "the device", "it"


@dataclass
class LinkedEntity:
    """An EntityMention after registry resolution."""
    mention:    EntityMention
    entity:     Entity
    match_type: str   # "exact" | "alias" | "fuzzy" | "shorthand" | "new"
    similarity: float # 1.0 for exact, 0.0 for new


# ---------------------------------------------------------------------------
# Shorthand vocabulary
# Device type shorthands: maps surface form → EntityType
# ---------------------------------------------------------------------------

_SHORTHAND_MAP: dict[str, EntityType] = {
    # devices
    "my phone":       EntityType.DEVICE,
    "my laptop":      EntityType.DEVICE,
    "my computer":    EntityType.DEVICE,
    "my tablet":      EntityType.DEVICE,
    "my desktop":     EntityType.DEVICE,
    "my macbook":     EntityType.DEVICE,
    "my pc":          EntityType.DEVICE,
    "the phone":      EntityType.DEVICE,
    "the laptop":     EntityType.DEVICE,
    "the device":     EntityType.DEVICE,
    "the machine":    EntityType.DEVICE,
    # people
    "my boss":        EntityType.PERSON,
    "my manager":     EntityType.PERSON,
    "my colleague":   EntityType.PERSON,
    "my friend":      EntityType.PERSON,
    "my co-founder":  EntityType.PERSON,
    # places
    "my office":      EntityType.PLACE,
    "my home":        EntityType.PLACE,
    "the office":     EntityType.PLACE,
    # projects
    "the project":    EntityType.PROJECT,
    "my project":     EntityType.PROJECT,
    "the app":        EntityType.PROJECT,
    "my app":         EntityType.PROJECT,
}


# ---------------------------------------------------------------------------
# String similarity
# ---------------------------------------------------------------------------

def _token_overlap(a: str, b: str) -> float:
    """
    Jaccard similarity over word tokens.
    Fast, no external deps. Good enough for entity matching.
    """
    ta = set(re.findall(r"\w+", a.lower()))
    tb = set(re.findall(r"\w+", b.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _name_similarity(a: str, b: str) -> float:
    """
    Combines token overlap with a substring containment check.
    "Galaxy S22" vs "Samsung Galaxy S22" → high similarity despite one being a subset.
    """
    overlap = _token_overlap(a, b)
    al, bl  = a.lower(), b.lower()
    # Substring bonus: if one is contained in the other
    if al in bl or bl in al:
        overlap = max(overlap, 0.80)
    return overlap


# ---------------------------------------------------------------------------
# Entity Registry (in-memory, swappable for SQLite in Phase 3)
# ---------------------------------------------------------------------------

class EntityRegistry:
    """
    Stores all known entities and exposes find / create / update operations.

    The registry is intentionally simple — a list scan with similarity scoring.
    For the memory volumes Almond handles (thousands, not millions) this is fast
    enough. Replace with an indexed store if needed.
    """

    EXACT_THRESHOLD:  float = 1.0
    ALIAS_THRESHOLD:  float = 0.95
    FUZZY_THRESHOLD:  float = 0.85
    REVIEW_THRESHOLD: float = 0.60   # below this: new entity (no merge attempt)

    def __init__(self):
        self._entities: dict[str, Entity] = {}   # id → Entity

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def find_by_id(self, entity_id: str) -> Optional[Entity]:
        return self._entities.get(entity_id)

    def find_by_name(self, name: str,
                     entity_type: Optional[EntityType] = None) -> Optional[Entity]:
        """Return best-matching entity for a name, or None."""
        best_entity = None
        best_score  = 0.0

        for entity in self._entities.values():
            if entity_type and entity.type != entity_type:
                continue
            for known_name in entity.all_names():
                score = _name_similarity(name, known_name)
                if score > best_score:
                    best_score  = score
                    best_entity = entity

        if best_score >= self.REVIEW_THRESHOLD:
            return best_entity
        return None

    def find_best_match(self, name: str,
                        entity_type: Optional[EntityType] = None
                        ) -> tuple[Optional[Entity], float]:
        """Returns (best_entity, similarity_score)."""
        best_entity = None
        best_score  = 0.0

        for entity in self._entities.values():
            if entity_type and entity.type != entity_type:
                continue
            for known_name in entity.all_names():
                score = _name_similarity(name, known_name)
                if score > best_score:
                    best_score  = score
                    best_entity = entity

        return best_entity, best_score

    def find_by_type(self, entity_type: EntityType) -> list[Entity]:
        return [e for e in self._entities.values() if e.type == entity_type]

    def most_recent_of_type(self, entity_type: EntityType) -> Optional[Entity]:
        """Returns the most recently seen entity of a given type."""
        candidates = [e for e in self._entities.values()
                      if e.type == entity_type and e.last_seen]
        if not candidates:
            return None
        return max(candidates, key=lambda e: e.last_seen)

    def all_entities(self) -> list[Entity]:
        return list(self._entities.values())

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def create(self, name: str, entity_type: EntityType,
               first_seen: Optional[datetime] = None,
               memory_id: Optional[str] = None,
               needs_review: bool = False) -> Entity:
        entity = Entity(
            id=str(uuid.uuid4()),
            name=name,
            type=entity_type,
            aliases=set(),
            first_seen=first_seen,
            last_seen=first_seen,
            memory_ids=[memory_id] if memory_id else [],
            needs_review=needs_review,
            reference_count=1,
        )
        self._entities[entity.id] = entity
        logger.debug("EntityRegistry.create: %s (%s) id=%s", name, entity_type.value, entity.id)
        return entity

    def link_memory(self, entity_id: str, memory_id: str,
                    alias: Optional[str] = None,
                    timestamp: Optional[datetime] = None) -> Optional[Entity]:
        """
        Record that a memory references an entity.
        Optionally add a new alias and update last_seen.
        """
        entity = self._entities.get(entity_id)
        if not entity:
            return None
        if memory_id not in entity.memory_ids:
            entity.memory_ids.append(memory_id)
        if alias and alias.lower() != entity.name.lower():
            entity.aliases.add(alias)
        if timestamp:
            if not entity.last_seen or timestamp > entity.last_seen:
                entity.last_seen = timestamp
        entity.reference_count += 1
        return entity

    def merge(self, keep_id: str, discard_id: str) -> Optional[Entity]:
        """
        Merge discard entity into keep entity.
        Used by the consolidator when a duplicate is confirmed.
        """
        keep    = self._entities.get(keep_id)
        discard = self._entities.get(discard_id)
        if not keep or not discard:
            return None

        keep.aliases.update(discard.aliases)
        keep.aliases.add(discard.name)
        keep.memory_ids.extend(m for m in discard.memory_ids if m not in keep.memory_ids)
        keep.reference_count += discard.reference_count
        if discard.first_seen and (not keep.first_seen or discard.first_seen < keep.first_seen):
            keep.first_seen = discard.first_seen
        if discard.last_seen and (not keep.last_seen or discard.last_seen > keep.last_seen):
            keep.last_seen = discard.last_seen
        keep.needs_review = False

        del self._entities[discard_id]
        logger.info("EntityRegistry.merge: merged %s into %s", discard_id, keep_id)
        return keep

    def __len__(self):
        return len(self._entities)


# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

_EXTRACTION_PROMPT = """You are an entity extraction engine for a personal memory system.

Extract all named entities from the text. For each entity return its exact surface form and type.

ENTITY TYPES:
- person:       named person ("Ashmit", "my manager Sarah", "Elon Musk")
- device:       hardware / gadget ("Samsung Galaxy S22", "MacBook Pro", "my laptop")
- place:        location ("London", "my office", "IITB")
- project:      named project or work item ("Almond", "Project X", "the memory system")
- organization: company / institution ("Anthropic", "Google", "my university")
- concept:      abstract topic / technology ("machine learning", "dark mode", "Python")
- unknown:      does not fit above

RULES:
- Use the most specific form available ("Samsung Galaxy S22" not "phone")
- Include informal references ("my laptop" → device, "the project" → project)
- Do NOT invent entities not present in the text
- Exclude common stop-words ("I", "it", "that", "this")

TEXT:
"{text}"

Respond with ONLY a valid JSON array (no markdown, no preamble):
[
  {{"name": "Samsung Galaxy S22", "type": "device"}},
  {{"name": "Ashmit",             "type": "person"}}
]

If no entities found, return: []"""


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

class EntityExtractor:
    """
    Extracts entity mentions from memory text and links them to the EntityRegistry.

    Usage
    -----
    registry  = EntityRegistry()
    extractor = EntityExtractor(registry=registry, llm=my_llm)

    linked = extractor.extract_and_link(
        memory_id="m1",
        text="I bought a Samsung Galaxy S22 in January.",
        timestamp=datetime.now(),
    )
    # linked is a list of LinkedEntity objects
    # registry now contains a "Samsung Galaxy S22" DEVICE entity
    """

    def __init__(self,
                 registry: EntityRegistry,
                 llm=None,
                 heuristic_only: bool = False):
        self._registry       = registry
        self._llm            = llm
        self._heuristic_only = heuristic_only or (llm is None)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_and_link(self,
                         memory_id:  str,
                         text:       str,
                         timestamp:  Optional[datetime] = None,
                         ) -> list[LinkedEntity]:
        """
        Extract entities from text and link them to the registry.
        Returns a list of LinkedEntity objects (mention + resolved entity + match type).
        """
        ts = timestamp or datetime.now()

        # 1. Get raw mentions
        mentions = self._extract_mentions(text)

        # 2. Resolve each mention
        linked = []
        for mention in mentions:
            le = self._resolve(mention, memory_id, ts)
            if le:
                linked.append(le)

        return linked

    def get_entity_ids(self, linked: list[LinkedEntity]) -> list[str]:
        return [le.entity.id for le in linked]

    # ------------------------------------------------------------------
    # Mention extraction
    # ------------------------------------------------------------------

    def _extract_mentions(self, text: str) -> list[EntityMention]:
        """
        Extract raw entity mentions. LLM path first, heuristic fallback.
        """
        mentions: list[EntityMention] = []

        if not self._heuristic_only and self._llm:
            mentions = self._llm_extract(text)

        if not mentions:
            mentions = self._heuristic_extract(text)

        return mentions

    def _llm_extract(self, text: str) -> list[EntityMention]:
        prompt = _EXTRACTION_PROMPT.format(text=text[:800])
        try:
            raw = self._llm.complete(prompt, max_tokens=400)
        except Exception as e:
            logger.warning("EntityExtractor LLM call failed: %s", e)
            return []

        return self._parse_llm_response(raw)

    def _parse_llm_response(self, raw: str) -> list[EntityMention]:
        if not raw:
            return []

        text = raw.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$",           "", text)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if not match:
                return []
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return []

        if not isinstance(data, list):
            return []

        mentions = []
        for item in data:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            if not name or len(name) < 2:
                continue
            raw_type = str(item.get("type", "unknown")).lower()
            try:
                etype = EntityType(raw_type)
            except ValueError:
                etype = EntityType.UNKNOWN

            is_shorthand = name.lower() in _SHORTHAND_MAP
            mentions.append(EntityMention(
                surface_form=name,
                entity_type=etype,
                is_shorthand=is_shorthand,
            ))

        return mentions

    # Heuristic: simple patterns to catch the most common entity forms
    _HEURISTIC_ENTITY_PATTERNS: list[tuple[re.Pattern, EntityType]] = []

    @classmethod
    def _build_heuristic_patterns(cls):
        if cls._HEURISTIC_ENTITY_PATTERNS:
            return
        raw = [
            # Devices — specific model patterns (tightest first)
            # "Samsung Galaxy S22", "Samsung Galaxy A53", "Dell XPS 13", "HP Envy 15"
            (r"\b((?:Samsung|Apple|Dell|HP|Lenovo|Asus|Acer|Sony|LG|Huawei|OnePlus|Google|Microsoft)\s+(?:[A-Za-z]+\s+)?[A-Za-z]\d+[\w]*)",
             EntityType.DEVICE),
            (r"\b(MacBook(?:\s+(?:Pro|Air|Mini))?)\b",           EntityType.DEVICE),
            (r"\b(iPhone\s+\d+(?:\s+(?:Pro\s+Max|Pro|Max|Plus|Mini))?)\b", EntityType.DEVICE),
            (r"\b(iPad(?:\s+(?:Pro|Air|Mini))?(?:\s+\d+)?)\b", EntityType.DEVICE),
            (r"\b(Pixel\s+\d+[a-zA-Z]*)\b",                     EntityType.DEVICE),
            # "Dell XPS 13", "HP EliteBook 840", "Lenovo ThinkPad X1" — brand + name + number
            (r"\b((?:Dell|HP|Lenovo|Asus|Acer|MSI|Razer|Huawei|Surface)\s+[A-Za-z]+(?:\s+[A-Za-z]\d?)?\s+\d+[a-zA-Z]*)\b",
             EntityType.DEVICE),
            # Shorthand devices  
            (r"\b(my (?:phone|laptop|computer|tablet|desktop|pc|macbook))\b",
             EntityType.DEVICE),
            # Named projects / products — all-caps or TitleCase 2+ words
            # E.g. "Almond Lab", "Project X", "the Almond project"
            (r"\b(?:project|the)\s+([A-Z][a-zA-Z0-9]{2,})\b",   EntityType.PROJECT),
        ]
        cls._HEURISTIC_ENTITY_PATTERNS = [
            (re.compile(p, re.IGNORECASE), etype)
            for p, etype in raw
        ]

    def _heuristic_extract(self, text: str) -> list[EntityMention]:
        self._build_heuristic_patterns()
        mentions = []
        seen = set()

        # Shorthand check first
        for shorthand, etype in _SHORTHAND_MAP.items():
            if re.search(r'\b' + re.escape(shorthand) + r'\b', text, re.I):
                if shorthand not in seen:
                    seen.add(shorthand)
                    mentions.append(EntityMention(
                        surface_form=shorthand,
                        entity_type=etype,
                        is_shorthand=True,
                    ))

        # Pattern-based extraction
        for pattern, etype in self._HEURISTIC_ENTITY_PATTERNS:
            for m in pattern.finditer(text):
                name = m.group(1).strip()
                if name.lower() not in seen and len(name) >= 2:
                    seen.add(name.lower())
                    mentions.append(EntityMention(
                        surface_form=name,
                        entity_type=etype,
                        is_shorthand=False,
                    ))

        return mentions

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def _resolve(self, mention: EntityMention, memory_id: str,
                 timestamp: datetime) -> Optional[LinkedEntity]:
        """
        Resolve a mention to an entity in the registry.
        Creates a new entity if no match is found.
        """
        name  = mention.surface_form
        etype = mention.entity_type

        # --- Shorthand: resolve via type + recency ---
        if mention.is_shorthand:
            return self._resolve_shorthand(mention, memory_id, timestamp)

        # --- Look for best match in registry ---
        candidate, score = self._registry.find_best_match(name, etype)

        # Exact or alias match
        if score >= EntityRegistry.ALIAS_THRESHOLD:
            self._registry.link_memory(candidate.id, memory_id,
                                       alias=name, timestamp=timestamp)
            match_type = "exact" if score == 1.0 else "alias"
            return LinkedEntity(mention=mention, entity=candidate,
                                match_type=match_type, similarity=score)

        # Fuzzy match — link but mark as possible merge candidate
        if score >= EntityRegistry.FUZZY_THRESHOLD:
            self._registry.link_memory(candidate.id, memory_id,
                                       alias=name, timestamp=timestamp)
            candidate.needs_review = True
            return LinkedEntity(mention=mention, entity=candidate,
                                match_type="fuzzy", similarity=score)

        # Possible duplicate but below fuzzy threshold — create new, flag
        needs_review = (score >= EntityRegistry.REVIEW_THRESHOLD)
        new_entity = self._registry.create(
            name=name,
            entity_type=etype,
            first_seen=timestamp,
            memory_id=memory_id,
            needs_review=needs_review,
        )
        return LinkedEntity(mention=mention, entity=new_entity,
                            match_type="new", similarity=0.0)

    def _resolve_shorthand(self, mention: EntityMention, memory_id: str,
                            timestamp: datetime) -> Optional[LinkedEntity]:
        """
        Resolve shorthands like "my phone" to the most recently seen
        entity of that type.
        """
        etype = _SHORTHAND_MAP.get(mention.surface_form.lower(), mention.entity_type)
        recent = self._registry.most_recent_of_type(etype)

        if recent:
            self._registry.link_memory(recent.id, memory_id,
                                       alias=mention.surface_form, timestamp=timestamp)
            return LinkedEntity(mention=mention, entity=recent,
                                match_type="shorthand", similarity=0.9)

        # No entity of this type yet — create a placeholder
        new_entity = self._registry.create(
            name=mention.surface_form,
            entity_type=etype,
            first_seen=timestamp,
            memory_id=memory_id,
            needs_review=True,  # placeholder needs a real name later
        )
        return LinkedEntity(mention=mention, entity=new_entity,
                            match_type="new", similarity=0.0)