#!/usr/bin/env python3
"""
Football Assistant – CLI mode

Use the same brains as the web app, but in your terminal.

Examples:
  - One‑off question (basic engine):
      python "python/football assistant/cli.py" -q "show la liga table"

  - Interactive REPL:
      python "python/football assistant/cli.py"

  - Use the advanced intelligence engine:
      python "python/football assistant/cli.py" --engine advanced

Notes:
  - Set API keys in the .env as you do for the web app (FOOTBALL_DATA_KEY, etc.).
  - REPL supports simple commands: :quit, :reset
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from typing import Dict, List, Optional


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass


async def _ask_basic(question: str, history: List[Dict], context: Dict) -> str:
    # Import locally to avoid importing FastAPI app at module import of this file
    from working_main import process_basic_query
    return await process_basic_query(question, history, context)


async def _ask_advanced(question: str, conversation_id: str) -> str:
    # Use the intelligence engine directly (avoids needing the web app running)
    from llm_intelligence_engine import create_intelligence_engine
    coordinator = create_intelligence_engine()
    res = await coordinator.process_intelligent_query(question, conversation_id)
    return res.get("answer") or str(res)


async def run_once(args: argparse.Namespace) -> int:
    hist: List[Dict] = []
    ctx: Dict = {}
    q = args.question or ""
    if not q:
        print("No question provided (-q). Try interactive mode without -q.")
        return 1

    if args.engine == "advanced":
        out = await _ask_advanced(q, args.conversation_id or "cli_session")
    else:
        out = await _ask_basic(q, hist, ctx)
    print(out)
    return 0


async def run_repl(args: argparse.Namespace) -> int:
    print("Football Assistant CLI – type :quit to exit, :reset to clear context.")
    engine = args.engine
    hist: List[Dict] = []
    ctx: Dict = {}
    conv_id = args.conversation_id or "cli_session"

    while True:
        try:
            q = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in {":q", ":quit", ":exit"}:
            break
        if q.lower() == ":reset":
            hist.clear()
            ctx.clear()
            print("(context cleared)")
            continue

        if engine == "advanced":
            out = await _ask_advanced(q, conv_id)
        else:
            out = await _ask_basic(q, hist, ctx)
            # maintain a lightweight history for basic engine
            hist.append({"role": "user", "content": q})
            hist.append({"role": "assistant", "content": out})
            if len(hist) > 40:
                hist[:] = hist[-40:]

        print(out)

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    _load_dotenv_if_available()

    p = argparse.ArgumentParser(description="Football Assistant CLI")
    p.add_argument("-q", "--question", help="Ask a one‑off question and exit")
    p.add_argument("--engine", choices=["basic", "advanced"], default="basic",
                   help="Choose processing engine (default: basic)")
    p.add_argument("--conversation-id", default=None,
                   help="Conversation id (advanced engine / REPL)")

    args = p.parse_args(argv)

    if args.question:
        return asyncio.run(run_once(args))
    return asyncio.run(run_repl(args))


if __name__ == "__main__":
    sys.exit(main())

