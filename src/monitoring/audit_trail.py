from __future__ import annotations

import aiofiles
import json
import os
from datetime import datetime
from hashlib import sha256
import hmac
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict

from exceptions import AuditTrailError


class AuditEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    timestamp: datetime
    event_type: str
    data: Dict[str, Any]
    prev_hash: str
    hash: str


class AuditTrailRecorder:
    """Append-only audit trail with hash chaining."""

    def __init__(self, file_path: Optional[str] = None, secret: Optional[str] = None) -> None:
        self.file_path = file_path or os.getenv("AUDIT_LOG_PATH", "audit.log")
        self.secret = secret or os.getenv("AUDIT_SECRET_KEY")
        if not self.secret:
            raise AuditTrailError("Audit secret key not set")

    async def _compute_hash(self, content: str) -> str:
        return hmac.new(self.secret.encode(), content.encode(), sha256).hexdigest()

    async def _get_last_hash(self) -> str:
        try:
            async with aiofiles.open(self.file_path, "r") as f:
                lines = await f.readlines()
            if lines:
                last = AuditEntry.model_validate_json(lines[-1].strip())
                return last.hash
        except FileNotFoundError:
            return ""
        except Exception as exc:  # pragma: no cover - rare read failure
            raise AuditTrailError(f"Read failure: {exc}") from exc
        return ""

    async def record_event(self, event_type: str, data: Dict[str, Any]) -> None:
        prev_hash = await self._get_last_hash()
        base = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "data": data,
            "prev_hash": prev_hash,
        }
        body = json.dumps(base, sort_keys=True, default=str)
        base["hash"] = await self._compute_hash(prev_hash + body)
        try:
            async with aiofiles.open(self.file_path, "a") as f:
                await f.write(json.dumps(base, sort_keys=True) + "\n")
        except Exception as exc:  # pragma: no cover - file system failure
            raise AuditTrailError(f"Write failure: {exc}") from exc

    async def verify_chain(self) -> bool:
        prev_hash = ""
        try:
            async with aiofiles.open(self.file_path, "r") as f:
                async for line in f:
                    rec = AuditEntry.model_validate_json(line.strip())
                    body = json.dumps(
                        {
                            "timestamp": rec.timestamp.isoformat(),
                            "event_type": rec.event_type,
                            "data": rec.data,
                            "prev_hash": prev_hash,
                        },
                        sort_keys=True,
                        default=str,
                    )
                    expected = await self._compute_hash(prev_hash + body)
                    if rec.hash != expected or rec.prev_hash != prev_hash:
                        return False
                    prev_hash = rec.hash
        except FileNotFoundError:
            return True
        except Exception as exc:  # pragma: no cover - read failure
            raise AuditTrailError(f"Verification failure: {exc}") from exc
        return True
