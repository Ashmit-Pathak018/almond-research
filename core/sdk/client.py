"""
Project Almond V3 — Typed Python SDK Client
Lightweight, fully typed HTTP client for external agents and applications.
Zero dependency on Almond internal storage or retrieval subsystems.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import httpx

from core.sdk.exceptions import (
    AlmondAPIError,
    AlmondConnectionError,
    AlmondNotFoundError,
    AlmondValidationError,
    AlmondConflictError,
)
from core.sdk.models import (
    CandidateRecord,
    EntityRecord,
    HealthResult,
    MemoryCreateResult,
    MemoryRecord,
    QueryResult,
    QueryTraceRecord,
    TimelineEventRecord,
    TimelineResult,
)


class BaseSubClient:
    def __init__(self, client: "Almond"):
        self._client = client

    def _request(self, method: str, path: str, **kwargs) -> Any:
        return self._client._request(method, path, **kwargs)


class MemoryClient(BaseSubClient):
    """Client for /v3/memories operations."""

    def create(
        self,
        content: str,
        namespace: Optional[str] = None,
        event_time: Optional[float] = None,
        tag: str = "EPISODIC",
        importance_score: float = 5.0,
        keywords: Optional[List[str]] = None,
        summary: Optional[str] = None,
        sync: bool = False,
    ) -> MemoryCreateResult:
        """Create or queue a memory."""
        ns = namespace or self._client.default_namespace
        payload = {
            "content": content,
            "namespace": ns,
            "event_time": event_time,
            "tag": tag,
            "importance_score": importance_score,
            "keywords": keywords or [],
            "summary": summary,
            "sync": sync,
        }
        data = self._request("POST", "/v3/memories", json=payload)
        return MemoryCreateResult(
            memory_id=data["memory_id"],
            job_id=data.get("job_id"),
            status=data["status"],
            namespace=data["namespace"],
        )

    def get(self, memory_id: str) -> MemoryRecord:
        """Fetch a canonical memory block by ID."""
        data = self._request("GET", f"/v3/memories/{memory_id}")
        return MemoryRecord(
            id=data["id"],
            namespace_id=data["namespace_id"],
            content=data["content"],
            summary=data.get("summary"),
            tag=data["tag"],
            tier=data["tier"],
            state=data["state"],
            importance_score=data["importance_score"],
            p_eff=data["p_eff"],
            keywords=data.get("keywords", []),
            event_time=data.get("event_time"),
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            last_accessed_at=data["last_accessed_at"],
            access_count=data["access_count"],
        )

    def delete(self, memory_id: str) -> bool:
        """Delete a memory block and cascade clean derived indexes."""
        data = self._request("DELETE", f"/v3/memories/{memory_id}")
        return bool(data.get("deleted", False))


class EntityClient(BaseSubClient):
    """Client for /v3/entities operations."""

    def get(self, entity_id: str, namespace: Optional[str] = None) -> EntityRecord:
        """Fetch entity by ID with aliases and linked memories."""
        params = {}
        if namespace or self._client.default_namespace != "default":
            params["namespace"] = namespace or self._client.default_namespace
        data = self._request("GET", f"/v3/entities/{entity_id}", params=params)
        return EntityRecord(
            entity_id=data["entity_id"],
            namespace_id=data["namespace_id"],
            canonical_name=data["canonical_name"],
            entity_type=data["entity_type"],
            summary=data.get("summary"),
            aliases=data.get("aliases", []),
            linked_memory_ids=data.get("linked_memory_ids", []),
            created_at=data["created_at"],
            updated_at=data["updated_at"],
        )


class Almond:
    """
    Main Almond V3 Typed SDK Client.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        namespace: str = "default",
        timeout: float = 30.0,
        http_client: Optional[Any] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.default_namespace = namespace
        self.timeout = timeout
        self._http = http_client or httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
        )

        # Sub-clients
        self.memories = MemoryClient(self)
        self.entities = EntityClient(self)

    def query(
        self,
        query: str,
        namespace: Optional[str] = None,
        reference_time: Optional[float] = None,
        top_k: int = 10,
    ) -> QueryResult:
        """Execute multi-channel retrieval, reasoning, ranking, and context assembly."""
        ns = namespace or self.default_namespace
        payload = {
            "query": query,
            "namespace": ns,
            "reference_time": reference_time,
            "top_k": top_k,
        }
        data = self._request("POST", "/v3/query", json=payload)

        candidates = [
            CandidateRecord(
                memory_id=c["memory_id"],
                final_score=c["final_score"],
                rank=c["rank"],
                content=c.get("content"),
                tag=c.get("tag"),
                tier=c.get("tier"),
                source_channels=c.get("source_channels", []),
                event_time=c.get("event_time"),
            )
            for c in data.get("candidates", [])
        ]

        return QueryResult(
            query=data["query"],
            namespace=data["namespace"],
            context_text=data["context_text"],
            is_abstention=data["is_abstention"],
            confidence=data["confidence"],
            memory_ids=data.get("memory_ids", []),
            candidates=candidates,
            trace_id=data.get("trace_id"),
        )

    def query_trace(
        self,
        query: str,
        namespace: Optional[str] = None,
        reference_time: Optional[float] = None,
        top_k: int = 10,
    ) -> QueryTraceRecord:
        """Execute query and return the full RetrievalTrace."""
        ns = namespace or self.default_namespace
        payload = {
            "query": query,
            "namespace": ns,
            "reference_time": reference_time,
            "top_k": top_k,
        }
        data = self._request("POST", "/v3/query/trace", json=payload)

        return QueryTraceRecord(
            trace_id=data["trace_id"],
            query=data["query"],
            namespace_id=data["namespace_id"],
            reference_time=data["reference_time"],
            detected_intent=data["detected_intent"],
            intent_confidence=data["intent_confidence"],
            channels_run=data["channels_run"],
            candidates_per_channel=data["candidates_per_channel"],
            fusion_merged_count=data["fusion_merged_count"],
            temporal_constraints_detected=data["temporal_constraints_detected"],
            rejected_count=data["rejected_count"],
            rejected_candidates=data.get("rejected_candidates", []),
            ranking_weights_used=data["ranking_weights_used"],
            ranked_candidates=data.get("ranked_candidates", []),
            final_context_ids=data.get("final_context_ids", []),
            duration_ms=data["duration_ms"],
            created_at=data["created_at"],
        )

    def timeline(
        self,
        namespace: Optional[str] = None,
        limit: int = 50,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> TimelineResult:
        """Query chronological timeline events."""
        ns = namespace or self.default_namespace
        params = {"namespace": ns, "limit": limit}
        if start_time is not None:
            params["start_time"] = start_time
        if end_time is not None:
            params["end_time"] = end_time

        data = self._request("GET", "/v3/timeline", params=params)

        events = [
            TimelineEventRecord(
                event_id=e["event_id"],
                namespace_id=e["namespace_id"],
                title=e["title"],
                event_type=e["event_type"],
                start_time=e.get("start_time"),
                end_time=e.get("end_time"),
                confidence=e.get("confidence", 1.0),
                linked_memory_ids=e.get("linked_memory_ids", []),
            )
            for e in data.get("events", [])
        ]

        return TimelineResult(
            namespace=data["namespace"],
            events=events,
            total_count=data["total_count"],
        )

    def health(self) -> HealthResult:
        """Check Almond service health."""
        data = self._request("GET", "/v3/health")
        return HealthResult(
            status=data["status"],
            version=data["version"],
            components=data["components"],
            clock=data["clock"],
        )

    def _request(self, method: str, path: str, **kwargs) -> Any:
        try:
            resp = self._http.request(method, path, **kwargs)
        except httpx.ConnectError as e:
            raise AlmondConnectionError(f"Failed to connect to Almond service at '{self.base_url}': {e}") from e
        except httpx.TimeoutException as e:
            raise AlmondConnectionError(f"Request to Almond service timed out after {self.timeout}s: {e}") from e
        except Exception as e:
            raise AlmondConnectionError(f"HTTP communication error: {e}") from e

        if resp.is_success:
            return resp.json()

        # Handle errors
        status_code = resp.status_code
        try:
            body = resp.json()
            detail = body.get("detail", body)
            err_type = body.get("error", "api_error")
        except Exception:
            detail = resp.text
            err_type = "unknown_error"

        msg = f"Almond API Error {status_code}: {detail}"

        if status_code == 404:
            raise AlmondNotFoundError(msg, status_code=status_code, error_type=err_type, detail=detail)
        elif status_code in (400, 422):
            raise AlmondValidationError(msg, status_code=status_code, error_type=err_type, detail=detail)
        elif status_code == 409:
            raise AlmondConflictError(msg, status_code=status_code, error_type=err_type, detail=detail)
        else:
            raise AlmondAPIError(msg, status_code=status_code, error_type=err_type, detail=detail)

    def close(self) -> None:
        if hasattr(self._http, "close"):
            try:
                self._http.close()
            except Exception:
                pass

    def __enter__(self) -> "Almond":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
