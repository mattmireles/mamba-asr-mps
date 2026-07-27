# COORD.md — session coordination ledger

Append-only, newest at the bottom, one line per substantive prompt when its work
lands: `- [YYYY-MM-DD HH:MMZ] [session-or-lane] <what was asked> -> <what landed> | evidence: <exit code / commit / path / status>`.
Honest entries only: in-progress is "in progress", untested is "untested". Compact
to COORD-ARCHIVE.md at ~40 ledger lines. In a fable-director arrangement, lane
blackboards live beside this file as COORD-<LANE>.md; this file is the ship/main ledger.

## LEDGER
- [2026-07-27 01:27Z] [hook] COORD.md scaffolded by oracle-suite SessionStart
- [2026-07-27 02:4xZ] [main] Import + adapt kokoro-coreml CLAUDE.md and skills -> 20 skills in .claude/skills/, README/{Skills,Templates,Guides,Notes,Plans} scaffolded, CLAUDE.md merged (Core ML field guide Parts 1-5 + skills index), AGENTS.md mirrored. Key adaptation: repo has NO test suite, so every pytest gate rewritten to `train_CTC.py --epochs 1 --sanity`. Skipped botnet + audio-judge (no analog). No existing files moved (scripts/report_phase3.sh reads README/training-notes.md at runtime). | evidence: `pytest --collect-only` -> "no tests collected"; sanity gate exit 0 (loss 16.86, 4.51s); link check 268/268 resolve, 0 empty dirs; 20 SKILL.md frontmatter names match dirs
