"""
Assignment 11 — Audit Log (nhật ký điều tra).

Ghi lại mọi tương tác để phục vụ điều tra. Lớp này không tự chặn gì —
các lớp khác bắt tấn công; lớp này giúp xem lại được chuyện đã xảy ra.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Luôn trỏ tới <repo>/outputs/… (đúng cả khi chạy từ trong src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Bộ ghi audit không phụ thuộc framework (pipeline gọi trước/sau mỗi request)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        # request đang chờ output: key -> {"input", "start", "timestamp"}
        self._open: dict[str, dict] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Lưu input + thời điểm bắt đầu, theo khoá request_id (hoặc user_id)."""
        key = request_id or user_id
        self._open[key] = {
            "input": text,
            "start": time.perf_counter(),
            "timestamp": utc_now_iso(),
        }

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Lưu output, lớp đã chặn, độ trễ; thêm vào self.logs."""
        key = request_id or user_id
        pending = self._open.pop(key, None) or {}
        start = pending.get("start")
        latency_ms = (time.perf_counter() - start) * 1000 if start else None

        self.logs.append({
            "request_id": request_id,
            "user_id": user_id,
            "timestamp": pending.get("timestamp", utc_now_iso()),
            "input": pending.get("input"),
            "output": text,
            "blocked": blocked,
            "layer": layer,
            "latency_ms": round(latency_ms, 1) if latency_ms is not None else None,
        })

    def export_json(self, filepath: str | None = None):
        """Ghi log ra đĩa (mảng JSON), mặc định vào ``outputs/`` ở gốc repo."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.logs, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
