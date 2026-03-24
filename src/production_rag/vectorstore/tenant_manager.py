"""Weaviate tenant lifecycle manager.

Tenants are created per research corpus / organization.
Operations are idempotent — creating an existing tenant is a no-op.
"""

from __future__ import annotations

import asyncio
from enum import Enum

import structlog
import weaviate
from weaviate.classes.tenants import Tenant, TenantActivityStatus

from production_rag.vectorstore.schema import COLLECTION_NAME
from production_rag.vectorstore.weaviate_client import WeaviateClient, get_weaviate_client

log = structlog.get_logger(__name__)


class TenantState(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"   # data on disk, not queryable
    OFFLOADED = "offloaded" # data moved to cold storage


_STATUS_MAP = {
    TenantState.ACTIVE: TenantActivityStatus.ACTIVE,
    TenantState.INACTIVE: TenantActivityStatus.INACTIVE,
    TenantState.OFFLOADED: TenantActivityStatus.OFFLOADED,
}


class TenantManager:
    """Manages Weaviate tenant lifecycle for ScientificChunk collection."""

    def __init__(self, client: WeaviateClient) -> None:
        self._client = client

    def _raw(self) -> weaviate.WeaviateClient:
        return self._client._get_client()  # noqa: SLF001

    async def create(self, tenant_id: str) -> None:
        """Create tenant if it does not exist (idempotent)."""
        raw = self._raw()

        def _create() -> None:
            collection = raw.collections.get(COLLECTION_NAME)
            existing = {t.name for t in collection.tenants.get().values()}
            if tenant_id in existing:
                log.debug("tenant.exists", tenant_id=tenant_id)
                return
            collection.tenants.create([Tenant(name=tenant_id)])
            log.info("tenant.created", tenant_id=tenant_id)

        await asyncio.get_event_loop().run_in_executor(None, _create)

    async def delete(self, tenant_id: str) -> None:
        """Delete tenant and all its data. DESTRUCTIVE."""
        raw = self._raw()

        def _delete() -> None:
            collection = raw.collections.get(COLLECTION_NAME)
            collection.tenants.remove([Tenant(name=tenant_id)])
            log.warning("tenant.deleted", tenant_id=tenant_id)

        await asyncio.get_event_loop().run_in_executor(None, _delete)

    async def set_state(self, tenant_id: str, state: TenantState) -> None:
        """Transition tenant to a new activity state."""
        raw = self._raw()
        weaviate_status = _STATUS_MAP[state]

        def _update() -> None:
            collection = raw.collections.get(COLLECTION_NAME)
            collection.tenants.update(
                [Tenant(name=tenant_id, activity_status=weaviate_status)]
            )
            log.info("tenant.state_changed", tenant_id=tenant_id, state=state)

        await asyncio.get_event_loop().run_in_executor(None, _update)

    async def list_tenants(self) -> list[dict[str, str]]:
        """Return list of {name, state} for all tenants."""
        raw = self._raw()

        def _list() -> list[dict[str, str]]:
            collection = raw.collections.get(COLLECTION_NAME)
            return [
                {"name": t.name, "state": t.activity_status.value}
                for t in collection.tenants.get().values()
            ]

        return await asyncio.get_event_loop().run_in_executor(None, _list)

    async def exists(self, tenant_id: str) -> bool:
        tenants = await self.list_tenants()
        return any(t["name"] == tenant_id for t in tenants)


async def get_tenant_manager() -> TenantManager:
    client = await get_weaviate_client()
    return TenantManager(client)
