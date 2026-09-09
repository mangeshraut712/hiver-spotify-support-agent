"""Trivial and simple baselines sharing AgentOutput schema."""

from __future__ import annotations

from typing import Optional, Sequence

from spotify_agent.intents import keyword_intent, should_escalate_heuristic
from spotify_agent.retrieve import ReplyRetriever
from spotify_agent.schemas import AgentOutput


TEMPLATE_REPLY = (
    "Hi there! Thanks for reaching out. Please DM us more details so we can look into this."
)


class TrivialBaseline:
    """Always other + template + escalate."""

    def run(
        self,
        customer_text: str,
        exclude_pair_ids: Optional[Sequence[str]] = None,
    ) -> AgentOutput:
        _ = customer_text, exclude_pair_ids
        return AgentOutput(
            intent="other",
            reply=TEMPLATE_REPLY,
            decision="escalate",
            reason="Trivial baseline always escalates.",
            confidence=0.0,
            retrieved_ids=[],
        )


class SimpleBaseline:
    """Keyword intent + nearest historical reply + heuristic escalate."""

    def __init__(self, retriever: ReplyRetriever, k: int = 1) -> None:
        self.retriever = retriever
        self.k = k

    def run(
        self,
        customer_text: str,
        exclude_pair_ids: Optional[Sequence[str]] = None,
    ) -> AgentOutput:
        intent = keyword_intent(customer_text)
        hits = self.retriever.search(
            customer_text, k=self.k, exclude_pair_ids=exclude_pair_ids
        )
        if hits:
            reply = hits[0].brand_reply_text
            retrieved_ids = [hits[0].pair_id]
            confidence = min(max(hits[0].score, 0.0), 1.0)
        else:
            reply = TEMPLATE_REPLY
            retrieved_ids = []
            confidence = 0.1

        escalate = should_escalate_heuristic(customer_text, intent)
        return AgentOutput(
            intent=intent,
            reply=reply,
            decision="escalate" if escalate else "auto",
            reason=(
                "Keyword intent + escalate heuristics"
                if escalate
                else "Keyword intent; no escalate triggers"
            ),
            confidence=confidence,
            retrieved_ids=retrieved_ids,
        )
