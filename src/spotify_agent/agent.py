"""LLM agent that classifies, drafts, and escalates."""

from __future__ import annotations

import os
from typing import Optional, Sequence

from spotify_agent.intents import (
    INTENT_DESCRIPTIONS,
    INTENTS,
    rule_based_intent,
    should_escalate_heuristic,
)
from spotify_agent.llm import agent_model, chat_json
from spotify_agent.paths import PROMPTS_DIR
from spotify_agent.reply_compose import compose_reply
from spotify_agent.retrieve import ReplyRetriever, RetrievedExample
from spotify_agent.schemas import AgentOutput, parse_agent_output


def _load_system_prompt() -> str:
    return (PROMPTS_DIR / "agent_system.txt").read_text()


def _format_examples(examples: Sequence[RetrievedExample]) -> str:
    blocks = []
    for i, ex in enumerate(examples, start=1):
        blocks.append(
            f"Example {i} (score={ex.score:.3f}, id={ex.pair_id}):\n"
            f"Customer: {ex.customer_text}\n"
            f"SpotifyCares: {ex.brand_reply_text}"
        )
    return "\n\n".join(blocks) if blocks else "(no exemplars)"


def build_user_prompt(
    customer_text: str,
    examples: Sequence[RetrievedExample],
    locked_intent: str,
) -> str:
    intent_lines = "\n".join(
        f"- {name}: {INTENT_DESCRIPTIONS[name]}" for name in INTENTS
    )
    return (
        f"Intent labels:\n{intent_lines}\n\n"
        f"LOCKED intent for this tweet (do not change): {locked_intent}\n\n"
        f"Retrieved historical resolutions:\n{_format_examples(examples)}\n\n"
        f"Incoming customer tweet:\n{customer_text}\n\n"
        "Draft a reply that names the issue type and gives a concrete next step.\n"
        "Return JSON with keys intent, reply, decision, reason, confidence.\n"
        f'Set "intent" exactly to "{locked_intent}".\n'
    )


class SupportAgent:
    """
    Hybrid agent:
    - Intent/escalate from stronger rule-based classifier.
    - Reply from issue-aware composer (default), optionally LLM-refined.
    Set AGENT_REPLY_MODE=retrieval|compose|llm (default compose).
    """

    def __init__(
        self,
        retriever: ReplyRetriever,
        k: int = 5,
        model: Optional[str] = None,
        use_cache: bool = True,
    ) -> None:
        self.retriever = retriever
        self.k = k
        self.model = model or agent_model()
        self.use_cache = use_cache
        self.system_prompt = _load_system_prompt()
        self.reply_mode = os.getenv("AGENT_REPLY_MODE", "compose").lower()

    def run(
        self,
        customer_text: str,
        exclude_pair_ids: Optional[Sequence[str]] = None,
    ) -> AgentOutput:
        examples = self.retriever.search(
            customer_text, k=self.k, exclude_pair_ids=exclude_pair_ids
        )
        intent, rule_esc = rule_based_intent(customer_text)
        decision = (
            "escalate"
            if (rule_esc or should_escalate_heuristic(customer_text, intent))
            else "auto"
        )

        sensitive = {
            "billing_charge",
            "account_security",
            "cancel_premium",
            "login_access",
            "family_duo_plan",
        }
        if intent in sensitive:
            decision = "escalate"
        # Playback with rule_esc True already covered; force escalate for broken downloads
        if intent == "playback_bug" and rule_esc:
            decision = "escalate"

        composed = compose_reply(intent, customer_text, examples, decision)
        reply = composed
        reason = "rules+composed_reply"
        confidence = 0.72

        if self.reply_mode == "retrieval" and examples:
            reply = examples[0].brand_reply_text
            reason = "rules+retrieval_copy"
            confidence = 0.6
        elif self.reply_mode == "llm":
            user = build_user_prompt(customer_text, examples, intent)
            try:
                raw = chat_json(
                    model=self.model,
                    system=self.system_prompt,
                    user=user,
                    temperature=0.1,
                    use_cache=self.use_cache,
                    cache_prefix="agent_v4",
                    max_retries=1,
                    rate_limit_sleep=False,
                )
                if isinstance(raw, list):
                    raw = raw[0] if raw and isinstance(raw[0], dict) else {}
                if not isinstance(raw, dict):
                    raw = {}
                draft = str(raw.get("reply", "")).strip()
                if draft:
                    reply = draft
                    reason = str(raw.get("reason", "llm draft with locked intent"))
                    try:
                        confidence = float(raw.get("confidence", 0.75))
                    except (TypeError, ValueError):
                        confidence = 0.75
                if raw.get("decision") == "escalate":
                    decision = "escalate"
            except Exception as exc:  # noqa: BLE001
                reason = f"composed_fallback:{type(exc).__name__}"
                confidence = 0.68

        confidence = min(max(confidence, 0.0), 1.0)
        if intent in sensitive:
            decision = "escalate"

        return parse_agent_output(
            {
                "intent": intent,
                "reply": reply,
                "decision": decision,
                "reason": reason,
                "confidence": confidence,
                "retrieved_ids": [ex.pair_id for ex in examples],
            }
        )
