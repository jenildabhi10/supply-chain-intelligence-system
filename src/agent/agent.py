"""
SupplyChainAgent — LLM-powered risk explanation using Groq + tool calling.

Architecture:
  1. User submits a natural-language question.
  2. Agent sends message + tool schemas to Groq (Llama 3.3 70B).
  3. Groq decides which tools to call (get_risk_card, get_active_signals, etc.).
  4. Agent executes tools, appends results to message history.
  5. Groq synthesises a final answer grounded strictly in tool results.
  6. Agent returns AgentResponse with answer + metadata.

Evidence-grounding contract:
  The system prompt forbids the LLM from adding any information not present
  in the tool results. If data is missing, the LLM must say so.

Fallback behaviour:
  If GROQ_API_KEY is absent or groq is not installed, the agent returns a
  structured text summary directly from the scoring system.
  Detectable via AgentResponse.llm_used == False.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import structlog

from config import settings
from src.agent.tools import TOOL_SCHEMAS, dispatch

# Sentinel: distinguishes "caller passed nothing" from "caller passed empty string"
_UNSET = object()

log = structlog.get_logger(__name__)

# ─── System prompt ────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a supply chain risk analyst assistant for a US port intelligence system.
Your job: answer questions about port disruption risk using ONLY the data returned by your tools.

STRICT RULES:
1. Always call the relevant tools BEFORE generating any answer.
2. Base your entire answer on tool results. Do NOT add external knowledge or speculation.
3. When citing a risk score, signal, or SHAP driver — reference the specific value from the tool result.
4. If tool results are empty or show no signals, state "No signals detected in the data."
5. Be concise and actionable. Analysts need clear, fast answers.
6. Always specify the port name (not just its ID) in your answer.
7. End with recommended actions when risk tier is medium, high, or critical.

Available ports: LA/LB, NY/NJ, Savannah, Seattle, Houston, Charleston, Norfolk, Oakland, Miami, Baltimore.\
"""


# ─── Response model ───────────────────────────────────────────────────────────

@dataclass
class AgentResponse:
    """The agent's complete response to a user query."""
    answer:          str
    tool_calls_made: list[str] = field(default_factory=list)
    llm_used:        bool = True
    model:           str | None = None
    error:           str | None = None


# ─── Agent ────────────────────────────────────────────────────────────────────

class SupplyChainAgent:
    """
    LLM agent that answers supply chain risk questions using Groq + tool calling.

    Usage:
        agent = SupplyChainAgent()
        response = agent.chat("What is the current risk at Los Angeles port?")
        print(response.answer)
    """

    def __init__(
        self,
        api_key:         object = _UNSET,   # pass "" to force fallback, omit to read from env
        model:           str = "llama-3.3-70b-versatile",
        max_tool_rounds: int = 5,
    ) -> None:
        self._model        = model
        self._max_rounds   = max_tool_rounds
        # Use env key only when caller did not pass anything explicitly
        self._api_key      = settings.GROQ_API_KEY if api_key is _UNSET else api_key
        self._client       = None

        if self._api_key:
            try:
                from groq import Groq
                self._client = Groq(api_key=self._api_key)
                log.info("agent_initialized", model=model)
            except ImportError:
                log.warning("groq_not_installed", msg="pip install groq to enable LLM features")
        else:
            log.warning("agent_no_api_key", msg="GROQ_API_KEY not set — using fallback mode")

    # ── Public API ────────────────────────────────────────────────────────────

    def chat(self, user_message: str) -> AgentResponse:
        """
        Answer a supply chain risk question.

        Uses Groq tool-calling loop when available; falls back to a structured
        text summary from the scoring engine when GROQ_API_KEY is not set.
        """
        if self._client is None:
            return self._fallback_response(user_message)
        try:
            return self._run_tool_loop(user_message)
        except Exception as exc:
            return self._handle_error(exc, user_message)

    # ── Internal: tool-calling loop ───────────────────────────────────────────

    def _run_tool_loop(self, user_message: str) -> AgentResponse:
        messages: list[dict] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ]
        tool_calls_made: list[str] = []

        for _ in range(self._max_rounds):
            response = self._client.chat.completions.create(
                model       = self._model,
                messages    = messages,
                tools       = TOOL_SCHEMAS,
                tool_choice = "auto",
                max_tokens  = 512,
            )

            choice = response.choices[0]

            if choice.finish_reason == "stop":
                return AgentResponse(
                    answer          = choice.message.content or "",
                    tool_calls_made = tool_calls_made,
                    llm_used        = True,
                    model           = self._model,
                )

            if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
                # Append assistant's tool-call turn
                messages.append({
                    "role":    "assistant",
                    "content": choice.message.content,  # may be None
                    "tool_calls": [
                        {
                            "id":   tc.id,
                            "type": "function",
                            "function": {
                                "name":      tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in choice.message.tool_calls
                    ],
                })

                for tc in choice.message.tool_calls:
                    tool_name = tc.function.name
                    try:
                        tool_args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        tool_args = {}

                    tool_calls_made.append(tool_name)
                    result_str = dispatch(tool_name, tool_args)
                    log.info("agent_tool_called", tool=tool_name)

                    messages.append({
                        "role":         "tool",
                        "tool_call_id": tc.id,
                        "content":      result_str,
                    })

            else:
                # Unexpected finish_reason — treat any content as final answer
                return AgentResponse(
                    answer          = choice.message.content or "",
                    tool_calls_made = tool_calls_made,
                    llm_used        = True,
                    model           = self._model,
                )

        # Exceeded max rounds
        last_answer = next(
            (m.get("content", "") for m in reversed(messages)
             if m.get("role") == "assistant" and m.get("content")),
            "Maximum tool-call rounds reached without a final answer.",
        )
        return AgentResponse(
            answer          = last_answer,
            tool_calls_made = tool_calls_made,
            llm_used        = True,
            model           = self._model,
            error           = "max_tool_rounds_exceeded",
        )

    # ── Error handling ────────────────────────────────────────────────────────

    def _handle_error(self, exc: Exception, user_message: str) -> AgentResponse:
        error_str = str(exc)
        log.error("agent_chat_error", error=error_str[:200])

        if "429" in error_str or "rate_limit_exceeded" in error_str:
            match = re.search(r"try again in ([^\.']+)", error_str, re.IGNORECASE)
            retry_hint = f" Resets in **{match.group(1).strip()}**." if match else ""
            fallback = self._fallback_response(user_message)
            fallback.answer = (
                f"Groq daily token limit reached (100k/day on free tier).{retry_hint}"
                f" Showing structured summary instead:\n\n{fallback.answer}"
            )
            fallback.error = "rate_limit_exceeded"
            return fallback

        if "400" in error_str or "tool_use_failed" in error_str:
            # LLM generated malformed tool-call JSON — retry once without tools
            log.warning("agent_tool_call_failed", msg="Retrying without tool schemas")
            try:
                resp = self._client.chat.completions.create(
                    model      = self._model,
                    messages   = [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user",   "content": user_message},
                    ],
                    max_tokens = 512,
                )
                return AgentResponse(
                    answer   = resp.choices[0].message.content or "",
                    llm_used = True,
                    model    = self._model,
                    error    = "tool_call_fallback",
                )
            except Exception:
                pass

        return AgentResponse(
            answer   = f"Agent error — {type(exc).__name__}: {error_str[:200]}",
            llm_used = False,
            error    = error_str[:200],
        )

    # ── Fallback: no Groq key ─────────────────────────────────────────────────

    def _fallback_response(self, user_message: str) -> AgentResponse:  # noqa: ARG002
        """Return a structured summary when Groq API is unavailable."""
        from src.agent.tools import get_all_port_scores

        try:
            result    = get_all_port_scores()
            summaries = result.get("port_summaries", [])
            if not summaries:
                return AgentResponse(
                    answer   = "No port risk data available (no Bronze/Gold data ingested yet).",
                    llm_used = False,
                )

            lines = ["**Current port risk summary:**\n"]
            lines.append("Top 3 ports by current risk score:")
            for s in summaries[:3]:
                lines.append(
                    f"  • {s['port_name']}: score={s['risk_score']:.1f} "
                    f"({s['risk_tier']}), confidence={s['confidence_score']:.0f}%, "
                    f"signals={s['n_signals']}"
                )
            return AgentResponse(answer="\n".join(lines), llm_used=False)

        except Exception as exc:
            return AgentResponse(
                answer   = f"Fallback scoring error: {exc}",
                llm_used = False,
                error    = str(exc),
            )
