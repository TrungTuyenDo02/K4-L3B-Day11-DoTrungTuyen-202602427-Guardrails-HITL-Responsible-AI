"""
Assignment 11 — Giám sát & cảnh báo (Monitoring & Alerts).

Theo dõi tỉ lệ bị chặn, số lần chạm rate limit, tỉ lệ judge đánh giá không an toàn.
Phát cảnh báo khi vượt ngưỡng.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


def default_metrics_path() -> str:
    """Luôn trỏ tới <repo>/outputs/… (đúng cả khi chạy từ trong src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "metrics.json")


@dataclass
class Alert:
    metric: str
    value: float
    threshold: float
    message: str


@dataclass
class MonitoringAlert:
    """Gom bộ đếm từ pipeline và phát cảnh báo."""

    block_rate_threshold: float = 0.5
    rate_limit_hit_threshold: int = 5
    judge_fail_rate_threshold: float = 0.3
    alerts: list[Alert] = field(default_factory=list)

    # Bộ đếm — pipeline cập nhật sau mỗi request
    total_requests: int = 0
    blocked_requests: int = 0
    rate_limit_hits: int = 0
    judge_checks: int = 0
    judge_fails: int = 0

    def check_metrics(self) -> list[Alert]:
        """Tính các tỉ lệ, tạo Alert khi vượt ngưỡng."""
        stats = self.snapshot()
        self.alerts = []  # tính lại từ đầu mỗi lần -> không bị trùng alert

        if stats["block_rate"] > self.block_rate_threshold:
            self.alerts.append(Alert(
                metric="block_rate",
                value=stats["block_rate"],
                threshold=self.block_rate_threshold,
                message="Tỉ lệ bị chặn cao — có thể đang bị tấn công hàng loạt hoặc chặn nhầm.",
            ))
        if self.rate_limit_hits > self.rate_limit_hit_threshold:
            self.alerts.append(Alert(
                metric="rate_limit_hits",
                value=self.rate_limit_hits,
                threshold=self.rate_limit_hit_threshold,
                message="Nhiều request chạm rate limit — có thể bị spam / tấn công làm tốn chi phí.",
            ))
        if self.judge_checks and stats["judge_fail_rate"] > self.judge_fail_rate_threshold:
            self.alerts.append(Alert(
                metric="judge_fail_rate",
                value=stats["judge_fail_rate"],
                threshold=self.judge_fail_rate_threshold,
                message="Nhiều câu trả lời bị judge đánh giá là không an toàn.",
            ))

        for alert in self.alerts:
            print(f"[ALERT] {alert.metric}={alert.value:.2f} > {alert.threshold}: {alert.message}")
        return self.alerts

    def export_json(self, filepath: str | None = None):
        """Ghi metrics + cảnh báo ra JSON, mặc định vào ``outputs/`` ở gốc repo."""
        self.check_metrics()
        path = Path(filepath or default_metrics_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.snapshot(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return path

    def snapshot(self) -> dict:
        block_rate = (
            self.blocked_requests / self.total_requests
            if self.total_requests
            else 0.0
        )
        judge_fail_rate = (
            self.judge_fails / self.judge_checks if self.judge_checks else 0.0
        )
        return {
            "total_requests": self.total_requests,
            "blocked_requests": self.blocked_requests,
            "block_rate": block_rate,
            "rate_limit_hits": self.rate_limit_hits,
            "judge_checks": self.judge_checks,
            "judge_fails": self.judge_fails,
            "judge_fail_rate": judge_fail_rate,
            "alerts": [
                {
                    "metric": a.metric,
                    "value": a.value,
                    "threshold": a.threshold,
                    "message": a.message,
                }
                for a in self.alerts
            ],
        }
