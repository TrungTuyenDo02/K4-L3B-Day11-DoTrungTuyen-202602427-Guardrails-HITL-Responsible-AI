"""
Assignment 11 — Rate Limiter.

Giới hạn số request theo từng user bằng cửa sổ trượt (sliding window).
Chặn kiểu lạm dụng mà các lớp guardrail khác không xử lý
(spam / tấn công làm tốn chi phí LLM).
"""
from __future__ import annotations

from collections import defaultdict, deque
import time

from google.adk.plugins import base_plugin
from google.genai import types


class RateLimitPlugin(base_plugin.BasePlugin):
    """Chặn user gửi quá max_requests trong window_seconds giây."""

    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        super().__init__(name="rate_limiter")
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.user_windows: dict[str, deque] = defaultdict(deque)
        self.blocked_count = 0
        self.total_count = 0

    def _block_response(self, message: str) -> types.Content:
        return types.Content(
            role="model",
            parts=[types.Part.from_text(text=message)],
        )

    async def on_user_message_callback(self, *, invocation_context, user_message):
        """Trả về Content để chặn, hoặc None để cho qua."""
        self.total_count += 1
        user_id = getattr(invocation_context, "user_id", None) or "anonymous"
        now = time.time()
        window = self.user_windows[user_id]

        # 1. Bỏ các timestamp đã ra khỏi cửa sổ (cũ hơn now - window_seconds)
        while window and window[0] <= now - self.window_seconds:
            window.popleft()

        # 2. Đã đủ max_requests trong cửa sổ -> chặn, không gọi LLM
        if len(window) >= self.max_requests:
            wait = self.window_seconds - (now - window[0])
            self.blocked_count += 1
            return self._block_response(
                "Bạn đã gửi quá nhiều yêu cầu (vượt rate limit). "
                f"Vui lòng thử lại sau {wait:.0f} giây."
            )

        # 3. Còn quota -> ghi lại thời điểm, cho qua
        window.append(now)
        return None
