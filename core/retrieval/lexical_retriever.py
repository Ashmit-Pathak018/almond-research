"""
Project Almond V3 — Lexical Retriever
SQLite FTS5-backed lexical and keyword retrieval.
Searches durable ground-truth SQLite data while strictly respecting namespace isolation.
"""

from __future__ import annotations
import json
import logging
import re
import sqlite3
from typing import List, Optional, Set, Any, Dict

from core.retrieval.contracts import RetrievalCandidate, RetrievalChannel, RetrievalQuery

logger = logging.getLogger(__name__)

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "at", "from",
    "by", "for", "with", "about", "against", "between", "into", "through", "during",
    "before", "after", "above", "below", "to", "in", "on", "off", "over", "under",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do",
    "does", "did", "can", "could", "should", "would", "will", "i", "you", "he",
    "she", "it", "we", "they", "my", "your", "his", "her", "their", "our", "what",
    "which", "who", "whom", "this", "that", "these", "those", "am", "so", "as"
}


class LexicalRetriever:
    """
    Retrieves candidates via SQLite FTS5 lexical search.
    Guarantees search capability even when vector indices (Chroma) are offline.
    """

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def retrieve(self, query: RetrievalQuery, top_k: Optional[int] = None) -> List[RetrievalCandidate]:
        k = top_k or query.top_k
        raw_tokens = re.findall(r"\w+", query.query_text.lower())
        tokens = [t for t in raw_tokens if t not in STOPWORDS and len(t) > 1]

        if not tokens:
            return []

        # Attempt FTS5 retrieval first
        candidates = self._retrieve_fts5(query, tokens, k)
        if candidates is not None:
            return candidates

        # Fallback to token/LIKE matching if FTS5 table is not initialized
        return self._retrieve_like_fallback(query, tokens, k)

    def _retrieve_fts5(
        self,
        query: RetrievalQuery,
        tokens: List[str],
        top_k: int
    ) -> Optional[List[RetrievalCandidate]]:
        """
        Executes query against SQLite FTS5 virtual table memory_blocks_fts.
        Returns None if table is absent or error occurs.
        """
        # Build sanitized FTS5 expression
        # Join tokens with OR, with prefix wildcards for stemming tolerance
        sanitized_tokens = [re.sub(r"[^\w]", "", t) for t in tokens if re.sub(r"[^\w]", "", t)]
        if not sanitized_tokens:
            return []

        fts_clause = " OR ".join(f'"{t}"*' for t in sanitized_tokens)
        # Also check for exact phrase if multi-token
        exact_phrase = " ".join(sanitized_tokens)
        if len(sanitized_tokens) > 1:
            fts_query = f'("{exact_phrase}") OR ({fts_clause})'
        else:
            fts_query = fts_clause

        sql = """
            SELECT f.id, bm25(memory_blocks_fts) AS bm25_score,
                   m.content, m.tag, m.tier, m.importance_score, m.keywords,
                   m.event_time, m.created_at, m.last_accessed_at, m.access_count
            FROM memory_blocks_fts f
            JOIN memory_blocks m ON f.id = m.id
            WHERE memory_blocks_fts MATCH ? AND f.namespace_id = ?
            ORDER BY bm25_score ASC
            LIMIT ?
        """

        try:
            rows = self._conn.execute(sql, (fts_query, query.namespace_id, top_k * 2)).fetchall()
        except sqlite3.OperationalError as e:
            logger.debug("LexicalRetriever FTS5 query failed (falling back to LIKE): %s", e)
            return None
        except Exception as e:
            logger.warning("Unexpected FTS5 query error: %s", e)
            return None

        if not rows:
            return []

        candidates: List[RetrievalCandidate] = []
        token_set = set(sanitized_tokens)

        for row in rows:
            mid = row["id"]
            raw_bm25 = float(row["bm25_score"]) if row["bm25_score"] is not None else 0.0
            
            # SQLite bm25() returns negative values where lower is better (e.g., -1.5 is better than -0.5).
            # Convert negative bm25 score to normalized similarity in [0.0, 1.0]:
            x = abs(raw_bm25)
            norm_bm25 = x / (1.0 + x)

            content_lower = row["content"].lower() if row["content"] else ""

            # Check exact term or exact phrase match for boosting
            phrase_boost = 0.2 if exact_phrase in content_lower else 0.0
            overlap_count = sum(1 for t in token_set if t in content_lower)
            overlap_ratio = overlap_count / float(len(token_set))

            if len(token_set) >= 2 and overlap_count < 2 and overlap_ratio < 0.35:
                # Incidental single token overlap on multi-token query
                continue

            norm_score = min(1.0, norm_bm25 * 0.7 + overlap_ratio * 0.2 + phrase_boost)

            cand = RetrievalCandidate(
                memory_id=mid,
                content=row["content"],
                tag=row["tag"],
                tier=row["tier"],
                event_time=row["event_time"],
                last_accessed_at=row["last_accessed_at"],
                importance_score=row["importance_score"],
            )
            cand.add_channel_result(
                channel=RetrievalChannel.LEXICAL,
                raw_score=raw_bm25,
                normalized_score=norm_score,
                metadata={"match_type": "fts5", "overlap_ratio": overlap_ratio}
            )
            candidates.append(cand)

        candidates.sort(
            key=lambda c: c.normalized_scores.get(RetrievalChannel.LEXICAL.value, 0.0),
            reverse=True
        )
        return candidates[:top_k]

    def _retrieve_like_fallback(
        self,
        query: RetrievalQuery,
        tokens: List[str],
        top_k: int
    ) -> List[RetrievalCandidate]:
        """Fallback when FTS5 virtual table is not present."""
        conditions = []
        params: List[Any] = [query.namespace_id]

        for tok in tokens:
            conditions.append("(LOWER(content) LIKE ? OR LOWER(keywords) LIKE ?)")
            pat = f"%{tok}%"
            params.extend([pat, pat])

        where_clause = " AND namespace_id = ? AND (" + " OR ".join(conditions) + ")"
        sql = f"""
            SELECT id, content, tag, tier, importance_score, keywords, event_time,
                   created_at, last_accessed_at, access_count
            FROM memory_blocks
            WHERE 1=1 {where_clause}
            ORDER BY last_accessed_at DESC
            LIMIT 50
        """

        try:
            rows = self._conn.execute(sql, params).fetchall()
        except Exception as e:
            logger.warning("LexicalRetriever LIKE query error: %s", e)
            return []

        if not rows:
            return []

        candidates: List[RetrievalCandidate] = []
        token_set = set(tokens)

        for row in rows:
            content_lower = row["content"].lower() if row["content"] else ""
            try:
                kw_list = json.loads(row["keywords"]) if row["keywords"] else []
                kw_lower = " ".join([k.lower() for k in kw_list])
            except Exception:
                kw_lower = ""

            matched_count = 0
            for t in token_set:
                if t in content_lower or t in kw_lower:
                    matched_count += 1

            overlap_ratio = matched_count / float(len(token_set))
            if len(token_set) >= 2 and matched_count < 2 and overlap_ratio < 0.35:
                continue

            exact_boost = 0.2 if query.query_text.lower() in content_lower else 0.0
            norm_score = min(1.0, overlap_ratio + exact_boost)

            cand = RetrievalCandidate(
                memory_id=row["id"],
                content=row["content"],
                tag=row["tag"],
                tier=row["tier"],
                event_time=row["event_time"],
                last_accessed_at=row["last_accessed_at"],
                importance_score=row["importance_score"],
            )
            cand.add_channel_result(
                channel=RetrievalChannel.LEXICAL,
                raw_score=float(matched_count),
                normalized_score=norm_score,
                metadata={"match_type": "like_fallback"}
            )
            candidates.append(cand)

        candidates.sort(
            key=lambda c: c.normalized_scores.get(RetrievalChannel.LEXICAL.value, 0.0),
            reverse=True
        )
        return candidates[:top_k]
