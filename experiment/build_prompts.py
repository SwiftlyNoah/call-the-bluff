"""Render the trial prompts in experiment/prompts/ from experiment/prompts/parts/.

The rendered files are what an agent sees, word for word; run.py reads them, never the parts.
Run: uv run experiment/build_prompts.py
"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
PARTS = HERE / "prompts" / "parts"
OUT = HERE / "prompts"

GAMES = {
    "kuhn": ("Kuhn poker", "a small poker game"),
    "leduc": ("Leduc hold'em", "a small poker game with two betting rounds and a public card"),
    "redblack": (
        "a mini Red/Black game",
        "a small two-player variant of Red/Black, a hidden-information bidding game",
    ),
}
RULES = {"kuhn": "kuhn_rules.md", "leduc": "leduc_rules.md"}


def part(name: str) -> str:
    return (PARTS / name).read_text()


def render() -> list[Path]:
    footer = part("claim_footer.md")
    written = []
    for game, (title, blurb) in GAMES.items():
        body = part("file_body.md").replace("{GAME_TITLE}", title).replace("{GAME_BLURB}", blurb)
        written.append(OUT / f"{game}_file.md")
        written[-1].write_text(body + footer)
        if game in RULES:
            body = part("rules_body.md").replace("{GAME_TITLE}", title)
            body = body.replace("{RULES}", part(RULES[game]).rstrip("\n"))
            written.append(OUT / f"{game}_rules.md")
            written[-1].write_text(body + footer)
    return written


if __name__ == "__main__":
    for p in render():
        print(p.relative_to(HERE.parent))
