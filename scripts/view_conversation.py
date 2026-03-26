#!/usr/bin/env python3
"""Pretty-print tau2-bench conversations with thinking/summary highlighting."""

from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from pathlib import Path

# ANSI colors
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
MAGENTA = "\033[35m"
BLUE = "\033[34m"
WHITE = "\033[37m"

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
SUMMARY_RE = re.compile(r"<think_summary>(.*?)</think_summary>", re.DOTALL)

WIDTH = 100


def wrap(text: str, indent: str = "  ", width: int = WIDTH) -> str:
    lines = text.split("\n")
    wrapped = []
    for line in lines:
        if len(line) <= width:
            wrapped.append(indent + line)
        else:
            wrapped.extend(
                textwrap.wrap(
                    line, width=width, initial_indent=indent, subsequent_indent=indent
                )
            )
    return "\n".join(wrapped)


def format_thinking(content: str) -> str:
    """Extract and format thinking blocks from content."""
    parts = []
    last_end = 0

    for match in THINK_RE.finditer(content):
        # Text before this thinking block
        before = content[last_end : match.start()].strip()
        if before:
            parts.append(("text", before))

        thinking = match.group(1).strip()
        if thinking:
            parts.append(("thinking", thinking))
        last_end = match.end()

    for match in SUMMARY_RE.finditer(content):
        before = content[last_end : match.start()].strip()
        if before:
            parts.append(("text", before))

        summary = match.group(1).strip()
        if summary:
            parts.append(("summary", summary))
        last_end = match.end()

    # Remaining text
    remaining = content[last_end:].strip()
    if remaining:
        parts.append(("text", remaining))

    return parts


def print_separator(char: str = "─", width: int = WIDTH) -> None:
    print(f"{DIM}{char * width}{RESET}")


def print_message(index: int, msg: dict, show_full_thinking: bool = False) -> None:
    role = msg.get("role", "?")
    content = msg.get("content") or ""
    tool_calls = msg.get("tool_calls")

    # Role header
    role_colors = {
        "assistant": CYAN,
        "user": GREEN,
        "tool": YELLOW,
        "system": DIM,
    }
    color = role_colors.get(role, WHITE)
    print(f"\n{color}{BOLD}[{index}] {role.upper()}{RESET}")

    if role == "tool":
        # Tool results — show compactly
        name = msg.get("name", "")
        if name:
            print(f"  {DIM}tool: {name}{RESET}")
        if content:
            truncated = content[:500] + ("..." if len(content) > 500 else "")
            print(wrap(truncated, indent=f"  {YELLOW}"))
            print(RESET, end="")
        return

    if not content and not tool_calls:
        print(f"  {DIM}(empty){RESET}")
        return

    # Parse content for thinking/summary blocks
    if content:
        parts = format_thinking(content)
        for kind, text in parts:
            if kind == "thinking":
                if show_full_thinking:
                    print(f"  {MAGENTA}{DIM}┌─ THINKING ─────────────────────{RESET}")
                    print(wrap(text, indent=f"  {MAGENTA}{DIM}│ "))
                    print(
                        f"{RESET}  {MAGENTA}{DIM}└───────────────────────────────{RESET}"
                    )
                else:
                    # Show truncated
                    lines = text.split("\n")
                    preview = lines[0][:200]
                    tokens_approx = len(text) // 4
                    print(
                        f"  {MAGENTA}{DIM}┌─ THINKING ({tokens_approx} tok approx, {len(text)} chars) ──{RESET}"
                    )
                    print(
                        f"  {MAGENTA}{DIM}│ {preview}{'...' if len(text) > 200 else ''}{RESET}"
                    )
                    print(f"  {MAGENTA}{DIM}└───────────────────────────────{RESET}")
            elif kind == "summary":
                print(f"  {BLUE}{BOLD}┌─ THINK SUMMARY ────────────────{RESET}")
                print(wrap(text, indent=f"  {BLUE}│ "))
                print(f"{RESET}  {BLUE}{BOLD}└────────────────────────────────{RESET}")
            else:
                print(wrap(text))

    # Tool calls
    if tool_calls:
        for tc in tool_calls:
            name = tc.get("name", "?")
            args = tc.get("arguments", "")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, TypeError):
                    pass
            args_str = (
                json.dumps(args, indent=2) if isinstance(args, dict) else str(args)
            )
            if len(args_str) > 300:
                args_str = args_str[:300] + "..."
            print(f"  {RED}→ TOOL CALL: {name}{RESET}")
            print(wrap(args_str, indent=f"    {DIM}"))
            print(RESET, end="")


def print_simulation(sim: dict, show_full_thinking: bool = False) -> None:
    task_id = sim.get("task_id", "?")
    duration = sim.get("duration", 0)
    termination = sim.get("termination_reason", "?")
    reward_info = sim.get("reward_info", {})
    reward = reward_info.get("reward", "?") if reward_info else "?"
    trial = sim.get("trial", 0)
    messages = sim.get("messages", [])

    print_separator("═")
    print(f"{BOLD}Task: {task_id}{RESET}")
    print(
        f"Trial: {trial}  |  Messages: {len(messages)}  |  Duration: {duration:.1f}s  |  Termination: {termination}"
    )
    print_separator("═")

    for i, msg in enumerate(messages):
        print_message(i, msg, show_full_thinking=show_full_thinking)

    # Reward summary
    print()
    print_separator("─")
    reward_str = f"{reward:.4f}" if isinstance(reward, (int, float)) else str(reward)
    icon = "✅" if reward == 1.0 else "❌"
    print(f"{BOLD}Reward: {icon} {reward_str}{RESET}")

    # Action details
    if reward_info:
        actions = reward_info.get("action_info", {})
        if isinstance(actions, dict):
            for key, val in actions.items():
                if isinstance(val, dict) and "reward" in val:
                    act_icon = "✅" if val["reward"] >= 1.0 else "❌"
                    print(f"  {act_icon} {key}: {val.get('reward', '?')}")


def find_latest_results(base: Path) -> Path | None:
    dirs = sorted(
        base.glob("*/results.json"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    return dirs[0].parent if dirs else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="result dir or results.json path")
    parser.add_argument(
        "--sim", type=int, default=0, help="simulation index (default: 0)"
    )
    parser.add_argument(
        "--list", action="store_true", help="list all simulations in the result"
    )
    parser.add_argument(
        "--full-thinking",
        action="store_true",
        help="show full thinking blocks (not truncated)",
    )
    parser.add_argument(
        "--latest", action="store_true", help="show the most recent result"
    )
    args = parser.parse_args()

    results_base = Path("results/phase1")

    if args.path:
        path = Path(args.path)
        if path.is_file():
            results_path = path
        elif path.is_dir():
            results_path = path / "results.json"
        else:
            print(f"Not found: {path}", file=sys.stderr)
            sys.exit(1)
    elif args.latest:
        latest = find_latest_results(results_base)
        if not latest:
            print("No results found", file=sys.stderr)
            sys.exit(1)
        results_path = latest / "results.json"
        print(f"{DIM}Using: {results_path}{RESET}")
    else:
        # List all available
        print(f"{BOLD}Available result directories:{RESET}")
        for d in sorted(results_base.iterdir()):
            if (d / "results.json").exists():
                summary_path = d / "summary.json"
                if summary_path.exists():
                    s = json.loads(summary_path.read_text())
                    reward = s.get("full_reward_count", "?")
                    total = s.get("num_simulations", "?")
                    print(f"  {d.name}  ({reward}/{total} passed)")
                else:
                    print(f"  {d.name}")
        print(f"\nUsage: python {sys.argv[0]} <dir> [--sim N] [--full-thinking]")
        return

    data = json.loads(results_path.read_text())
    sims = data.get("simulations", [])

    if args.list:
        print(f"{BOLD}Simulations in {results_path.parent.name}:{RESET}")
        for i, sim in enumerate(sims):
            reward = sim.get("reward_info", {})
            r = reward.get("reward", "?") if reward else "?"
            icon = "✅" if r == 1.0 else "❌" if isinstance(r, (int, float)) else "?"
            task = sim.get("task_id", "?")[:80]
            dur = sim.get("duration", 0)
            msgs = len(sim.get("messages", []))
            print(f"  [{i}] {icon} r={r}  {msgs} msgs  {dur:.0f}s  {task}")
        return

    if args.sim >= len(sims):
        print(f"Simulation {args.sim} not found (have {len(sims)})", file=sys.stderr)
        sys.exit(1)

    print_simulation(sims[args.sim], show_full_thinking=args.full_thinking)


if __name__ == "__main__":
    main()
