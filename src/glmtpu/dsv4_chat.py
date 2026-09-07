"""DeepSeek-V4 chat encoding — port of DeepSeek's encoding/encoding_dsv4.py
(research/dsv4-encoding.py, the repo's official chat template; the HF repo
ships no chat_template.jinja).  Implements the subset needed for serving:
encode_messages (thinking_mode chat/thinking) and completion parsing that
splits reasoning (<think> blocks) for reasoning_content.
"""
from __future__ import annotations

from typing import Any, Dict, List

bos_token = "<｜begin▁of▁sentence｜>"
eos_token = "<｜end▁of▁sentence｜>"
thinking_start_token = "<think>"
thinking_end_token = "</think>"
dsml_token = "｜DSML｜"

USER_SP_TOKEN = "<｜User｜>"
ASSISTANT_SP_TOKEN = "<｜Assistant｜>"
LATEST_REMINDER_SP_TOKEN = "<｜latest_reminder｜>"

assistant_msg_template = "{reasoning}{content}{tool_calls}" + eos_token
assistant_msg_wo_eos_template = "{reasoning}{content}{tool_calls}"
thinking_template = "{reasoning_content}"


def _has_image_or_video(messages) -> bool:
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            for part in c:
                if isinstance(part, dict) and part.get("type") in (
                        "image_url", "image", "video", "video_url",
                        "input_image", "input_video"):
                    return True
    return False


def find_last_user_index(messages) -> int:
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            return i
    return -1


def _content_str(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(p.get("text", ""))
        return "\n".join(parts)
    return ""


def render_message(index, messages, thinking_mode, drop_thinking):
    msg = messages[index]
    role = msg.get("role")
    content = _content_str(msg.get("content", ""))

    if role == "system":
        prompt = content
    elif role in ("user", "developer", "latest_reminder"):
        prompt = content
        # tool_result blocks
        blocks = msg.get("content_blocks") or []
        tr = [b.get("content", "") for b in blocks
              if b.get("type") == "tool_result"]
        if tr:
            prompt += "".join(f"<tool_result>{t}</tool_result>" for t in tr)
    elif role == "assistant":
        rc = msg.get("reasoning_content") or ""
        thinking_part = ""
        last_user_idx = find_last_user_index(messages)
        if thinking_mode == "thinking" and not (
                drop_thinking and index < last_user_idx):
            thinking_part = thinking_template.format(
                reasoning_content=rc) + thinking_end_token
        if msg.get("wo_eos"):
            prompt = assistant_msg_wo_eos_template.format(
                reasoning=thinking_part, content=content, tool_calls="")
        else:
            prompt = assistant_msg_template.format(
                reasoning=thinking_part, content=content, tool_calls="")
    else:
        raise NotImplementedError(f"Unknown role: {role}")

    # transition tokens after the final message
    if index + 1 < len(messages) and messages[index + 1].get("role") \
            not in ["assistant", "latest_reminder"]:
        return prompt
    if messages[index].get("role") in ("user", "developer"):
        prompt += ASSISTANT_SP_TOKEN
        if not drop_thinking and thinking_mode == "thinking":
            prompt += thinking_start_token
        elif drop_thinking and thinking_mode == "thinking" \
                and index >= find_last_user_index(messages):
            prompt += thinking_start_token
        else:
            prompt += thinking_end_token
    return prompt


def encode_messages(messages: List[Dict[str, Any]], thinking_mode="chat",
                    drop_thinking=True, add_default_bos_token=True) -> str:
    """Main entry: OpenAI-style messages -> DSV4 prompt string."""
    prompt = bos_token if add_default_bos_token else ""
    last_user_idx = find_last_user_index(messages)
    for idx in range(len(messages)):
        prompt += render_message(idx, messages, thinking_mode,
                                 drop_thinking)
    return prompt


def parse_message_from_completion_text(text: str,
                                       thinking_mode="chat") -> Dict[str, Any]:
    """Split a completion into reasoning_content / content (DSV4 thinks:
    output starts inside a <think> block when thinking_mode='thinking')."""
    msg = {"role": "assistant", "content": "", "reasoning_content": ""}
    if thinking_mode == "thinking":
        if text.startswith(thinking_start_token):
            # everything up to </think> is reasoning
            rest = text[len(thinking_start_token):]
            if thinking_end_token in rest:
                rc, content = rest.split(thinking_end_token, 1)
                msg["reasoning_content"] = rc
                msg["content"] = content
                return msg
            msg["reasoning_content"] = rest
            return msg
        if thinking_end_token in text:
            rc, content = text.split(thinking_end_token, 1)
            msg["reasoning_content"] = rc
            msg["content"] = content
            return msg
    msg["content"] = text
    return msg
