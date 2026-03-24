"""Tenant management router.

POST   /tenants          — create tenant
DELETE /tenants/{id}     — delete tenant (DESTRUCTIVE)
GET    /tenants          — list all tenants
PATCH  /tenants/{id}     — update state (active/inactive/offloaded)
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

from production_rag.vectorstore.tenant_manager import TenantState

router = APIRouter(prefix="/tenants", tags=["tenants"])


class TenantCreate(BaseModel):
    tenant_id: str


class TenantStateUpdate(BaseModel):
    state: TenantState


@router.post("", status_code=201)
async def create_tenant(body: TenantCreate, request: Request) -> dict:  # type: ignore[type-arg]
    manager = request.app.state.tenant_manager
    await manager.create(body.tenant_id)
    return {"tenant_id": body.tenant_id, "status": "created"}


@router.get("")
async def list_tenants(request: Request) -> dict:  # type: ignore[type-arg]
    manager = request.app.state.tenant_manager
    tenants = await manager.list_tenants()
    return {"tenants": tenants}


@router.patch("/{tenant_id}")
async def update_tenant_state(
    tenant_id: str, body: TenantStateUpdate, request: Request
) -> dict:  # type: ignore[type-arg]
    manager = request.app.state.tenant_manager
    if not await manager.exists(tenant_id):
        raise HTTPException(status_code=404, detail="Tenant not found")
    await manager.set_state(tenant_id, body.state)
    return {"tenant_id": tenant_id, "state": body.state}


@router.delete("/{tenant_id}", status_code=204)
async def delete_tenant(tenant_id: str, request: Request) -> Response:
    manager = request.app.state.tenant_manager
    if not await manager.exists(tenant_id):
        raise HTTPException(status_code=404, detail="Tenant not found")
    await manager.delete(tenant_id)
    return Response(status_code=204)
