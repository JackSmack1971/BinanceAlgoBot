import json
import pytest

from src.monitoring.audit_trail import AuditTrailRecorder


@pytest.mark.asyncio
async def test_record_and_verify(tmp_path, monkeypatch):
    log_file = tmp_path / "audit.log"
    monkeypatch.setenv("AUDIT_SECRET_KEY", "testkey")
    recorder = AuditTrailRecorder(file_path=str(log_file))
    await recorder.record_event("ORDER", {"id": 1})
    assert await recorder.verify_chain()


@pytest.mark.asyncio
async def test_tamper_detection(tmp_path, monkeypatch):
    log_file = tmp_path / "audit.log"
    monkeypatch.setenv("AUDIT_SECRET_KEY", "testkey")
    recorder = AuditTrailRecorder(file_path=str(log_file))
    await recorder.record_event("ORDER", {"id": 1})
    await recorder.record_event("ORDER", {"id": 2})
    lines = log_file.read_text().splitlines()
    first = json.loads(lines[0])
    first["data"] = {"id": 99}
    lines[0] = json.dumps(first)
    log_file.write_text("\n".join(lines) + "\n")
    assert not await recorder.verify_chain()
