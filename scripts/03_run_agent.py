#!/usr/bin/env python3
"""Run agent or baselines on a single message or golden set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from spotify_agent.agent import SupportAgent
from spotify_agent.baselines import SimpleBaseline, TrivialBaseline
from spotify_agent.retrieve import ReplyRetriever


def get_system(name: str, retriever: ReplyRetriever):
    if name == "agent":
        return SupportAgent(retriever)
    if name == "simple":
        return SimpleBaseline(retriever)
    if name == "trivial":
        return TrivialBaseline()
    raise ValueError(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=["agent", "simple", "trivial"], default="agent")
    parser.add_argument("--text", required=True)
    parser.add_argument("--exclude-id", default=None)
    args = parser.parse_args()

    retriever = ReplyRetriever.load()
    system = get_system(args.system, retriever)
    exclude = [args.exclude_id] if args.exclude_id else None
    out = system.run(args.text, exclude_pair_ids=exclude)
    print(json.dumps(out.model_dump(), indent=2))


if __name__ == "__main__":
    main()
