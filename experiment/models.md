# Which Claude models are reachable (probed 2026-11-20)

Each id was tried once with a trivial prompt through this session's pool account:

```
claude-pool current -p --model <id> --output-format json --max-turns 1 "Reply with exactly: OK"
```

Script: `experiment/models_probe/probe.sh` (raw replies in `experiment/models_probe/<id>.json`,
stderr in `<id>.err`). "Served by" is the key of `modelUsage` in the JSON reply, which is the
model that actually answered. It is not always the one asked for, so the harness records it on
every trial (`agent.models_used` in `result.json`).

| `--model` id | Works | Served by | Probe cost (USD) | Note |
| --- | --- | --- | --- | --- |
| `claude-haiku-5-5` | yes | claude-haiku-5-5 | 0.0032 | cheapest; used for the pilot |
| `claude-sonnet-5-5` | yes | claude-sonnet-5-5 | 0.060 | |
| `claude-opus-5-5` | yes | claude-opus-5-5 | 0.121 | |
| `claude-fable-5-2` | unknown | | | refused three times by claude-pool (`<id>.err`, `fable-retry2.err`), not by the API: "<pool account> is at its leave point (90% of its 5-hour window)", although the same account served other models in the same minute. Retried once more by the reviewer (`fable-retry3.err`), same refusal, while `claude-haiku-5-5` was served on the same account a minute later, so the guard refuses Fable specifically. Retry on another lane before the matrix |
| `claude-sonnet-4-5` | yes | claude-sonnet-4-5 | 0.078 | |
| `claude-haiku-4-5` | yes | claude-haiku-4-5 | 0.026 | |
| `claude-opus-4-1` | NO (silently substituted) | claude-opus-5-5 | 0.0055 | the request succeeded but `modelUsage` names claude-opus-5-5: do not use this id as an "older Opus" arm |
| `claude-sonnet-4-20250514` | no | | 0 | "There's an issue with the selected model ... may not exist or you may not have access to it" |
| `claude-3-7-sonnet-latest` | no | | 0 | same error |
| `claude-3-5-haiku-latest` | no | | 0 | same error |
| `haiku` (alias) | yes | claude-haiku-5-5 | 0.0003 | |
| `sonnet` (alias) | yes | claude-sonnet-5-5 | 0.053 | |
| `opus` (alias) | yes | claude-opus-5-5 | 0.106 | |

The probe costs are dominated by Claude Code's own system prompt and cache writes, not by the
reply, so they say little about trial cost; see `DESIGN.md` for per-trial costs from the pilot.

Usable model arms for the matrix: two generations of each tier where available,
`claude-haiku-4-5`, `claude-haiku-5-5`, `claude-sonnet-4-5`, `claude-sonnet-5-5`,
`claude-opus-5-5` (no older Opus is reachable), and `claude-fable-5-2` if a retry works.
