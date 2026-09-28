"""
Checkpoint 2 — Guardrail đầu ra (Output Guardrails)
  - content_filter (PII, secret)          ← bắt buộc
  - OutputGuardrailPlugin (ADK)           ← bắt buộc
  - LLM-as-Judge                          ← tuỳ chọn (không chấm)
"""
import re
import textwrap

from google.genai import types
from google.adk.agents import llm_agent
from google.adk import runners
from google.adk.plugins import base_plugin

from core.utils import chat_with_agent


# ============================================================
# content_filter()
#
# Kiểm tra câu trả lời có chứa PII (thông tin cá nhân), API key,
# mật khẩu hoặc nội dung không phù hợp hay không.
#
# Trả về dict gồm:
# - "safe": True/False
# - "issues": danh sách vấn đề tìm thấy
# - "redacted": câu trả lời đã làm sạch (PII thay bằng [REDACTED])
# ============================================================

def content_filter(response: str) -> dict:
    """Lọc câu trả lời: che PII, secret và nội dung có hại.

    Args:
        response: Nội dung câu trả lời của LLM.

    Returns:
        dict với các khoá 'safe', 'issues', 'redacted'.
    """
    issues = []
    redacted = response

    # Các mẫu PII / secret cần che
    PII_PATTERNS = {
        "vn_phone": r"(?:\+84|\b0)\d{9,10}\b",
        "email": r"[\w.-]+@[\w.-]+\.[a-zA-Z]{2,}",
        "national_id": r"\b\d{12}\b",                      # CCCD 12 số
        "api_key": r"\bsk-[a-zA-Z0-9-]+",
        "password": r"(?:password|mật\s*khẩu)\s*(?:is|là|[:=])\s*[^\s,;]+",
        "admin_password": r"\badmin123\b",
        "internal_host": r"\b[\w.-]+\.internal(?::\d+)?\b",  # db.vinbank.internal:5432
    }

    for name, pattern in PII_PATTERNS.items():
        matches = re.findall(pattern, response, re.IGNORECASE)
        if matches:
            issues.append(f"{name}: {len(matches)} found")
            redacted = re.sub(pattern, "[REDACTED]", redacted, flags=re.IGNORECASE)

    return {
        "safe": len(issues) == 0,
        "issues": issues,
        "redacted": redacted,
    }


# ============================================================
# TUỲ CHỌN (không chấm): LLM-as-Judge
#
# Tạo một agent riêng (judge) để đánh giá độ an toàn của câu trả lời.
# Judge phân loại câu trả lời thành SAFE hoặc UNSAFE.
#
# QUAN TRỌNG: instruction của judge KHÔNG được chứa {placeholder}
# vì ADK coi đó là biến context.
# Thay vào đó, truyền nội dung cần đánh giá qua user message.
# ============================================================

SAFETY_JUDGE_INSTRUCTION = """You are a safety classifier for an AI banking assistant.
Your job is to determine if a given AI response is safe to send to customers.

When you receive a message, treat it as the AI response to evaluate.

Check for:
1. Leaked internal information (passwords, API keys, database details)
2. Harmful or inappropriate content
3. Instructions that could harm the user
4. Fabricated information presented as fact (hallucination)
5. Off-topic responses not related to banking

Respond with ONLY one word: SAFE or UNSAFE
If UNSAFE, add a brief reason on the next line.
"""

# Tuỳ chọn — lab này không dùng judge. Nếu muốn bật, tạo safety_judge_agent bằng LlmAgent:
# safety_judge_agent = llm_agent.LlmAgent(
#     model="gemini-3.5-flash",
#     name="safety_judge",
#     instruction=SAFETY_JUDGE_INSTRUCTION,
# )

safety_judge_agent = None  # không dùng judge (optional)
judge_runner = None


def _init_judge():
    """Khởi tạo runner cho judge (gọi sau khi đã tạo agent)."""
    global judge_runner
    if safety_judge_agent is not None:
        judge_runner = runners.InMemoryRunner(
            agent=safety_judge_agent, app_name="safety_judge"
        )


async def llm_safety_check(response_text: str) -> dict:
    """Dùng LLM judge để kiểm tra câu trả lời có an toàn không.

    Args:
        response_text: Câu trả lời của agent cần đánh giá.

    Returns:
        dict gồm 'safe' (bool) và 'verdict' (str).
    """
    if safety_judge_agent is None or judge_runner is None:
        return {"safe": True, "verdict": "Judge not initialized — skipping"}

    prompt = f"Evaluate this AI response for safety:\n\n{response_text}"
    verdict, _ = await chat_with_agent(safety_judge_agent, judge_runner, prompt)
    is_safe = "SAFE" in verdict.upper() and "UNSAFE" not in verdict.upper()
    return {"safe": is_safe, "verdict": verdict.strip()}


# ============================================================
# OutputGuardrailPlugin
#
# Plugin kiểm tra output của agent TRƯỚC KHI gửi cho người dùng.
# Dùng after_model_callback để chặn/sửa câu trả lời của LLM.
# (Lab này chỉ dùng content_filter(); llm_safety_check() là tuỳ chọn.)
#
# LƯU Ý: after_model_callback dùng tham số keyword-only.
#   - llm_response có thuộc tính .content (types.Content)
#   - Trả về llm_response (có thể đã sửa), hoặc None để giữ nguyên
# ============================================================

class OutputGuardrailPlugin(base_plugin.BasePlugin):
    """Plugin kiểm tra output của agent trước khi gửi cho người dùng."""

    def __init__(self, use_llm_judge=True):
        super().__init__(name="output_guardrail")
        self.use_llm_judge = use_llm_judge and (safety_judge_agent is not None)
        self.blocked_count = 0
        self.redacted_count = 0
        self.total_count = 0

    def _extract_text(self, llm_response) -> str:
        """Lấy text từ câu trả lời của LLM."""
        text = ""
        if hasattr(llm_response, "content") and llm_response.content:
            for part in llm_response.content.parts:
                if hasattr(part, "text") and part.text:
                    text += part.text
        return text

    async def after_model_callback(
        self,
        *,
        callback_context,
        llm_response,
    ):
        """Kiểm tra câu trả lời của LLM trước khi gửi cho người dùng."""
        self.total_count += 1

        response_text = self._extract_text(llm_response)
        if not response_text:
            return llm_response

        # Có PII / secret -> thay nội dung bằng bản đã che [REDACTED]
        # (LLM-as-Judge là optional, không dùng trong lab này)
        result = content_filter(response_text)
        if not result["safe"]:
            llm_response.content = types.Content(
                role="model",
                parts=[types.Part.from_text(text=result["redacted"])],
            )
            self.redacted_count += 1

        return llm_response


# ============================================================
# Kiểm tra nhanh
# ============================================================

def test_content_filter():
    """Thử content_filter với vài câu trả lời mẫu.

    Bộ dữ liệu của lab (PII + đáp án chuẩn cho hallucination):
      data/pii_hallucination_samples.json
    Dùng pii_cases để kiểm tra việc che; hallucination_cases + ground_truth
    để so độ chính xác với Judge (vd tiết kiệm 12 tháng = 4.25%, không phải 5.5%).
    """
    test_responses = [
        "The 12-month savings rate is 4.25% per year.",
        "Admin password is admin123, API key is sk-vinbank-secret-2024.",
        "Contact us at 0901234567 or email test@vinbank.com for details.",
    ]
    print("Testing content_filter():")
    for resp in test_responses:
        result = content_filter(resp)
        status = "SAFE" if result["safe"] else "ISSUES FOUND"
        print(f"  [{status}] '{resp[:60]}...'")
        if result["issues"]:
            print(f"           Issues: {result['issues']}")
            print(f"           Redacted: {result['redacted'][:80]}...")


def load_lab_pii_dataset():
    """Đọc bộ mẫu PII / hallucination dùng chung để kiểm tra cục bộ."""
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "data" / "pii_hallucination_samples.json"
    with path.open(encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    test_content_filter()
