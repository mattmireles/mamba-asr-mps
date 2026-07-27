---
name: create-skill
description: >-
  Guides authoring a new Agent Skill for this repo. Use when the user wants to
  create, write, or author a new skill, or asks about skill structure, best
  practices, or SKILL.md format. Do not use for editing an existing skill's
  content when the structure is already correct—just edit it.
---

# Create Skill

## Where skills live here

`.claude/skills/<skill-name>/SKILL.md` — that's it. This repo has **one** skill
tree, no `.cursor/` or `.agents/` mirrors to keep in sync. Do not create user-level
skills under `~/` unless the user explicitly asks for a personal skill.

```text
.claude/skills/<skill-name>/
├── SKILL.md              # required
├── reference.md          # optional — indexed repo paths
├── references/index.md   # optional — canonical-doc map (pattern used by the workflow skills)
└── examples.md           # optional
```

## Before you begin

Gather from the user, or infer from the conversation:

1. **Purpose and scope** — what task or workflow?
2. **Trigger scenarios** — when should the agent reach for this automatically?
3. **Key domain knowledge** — what does the agent *not* already know?
4. **Output format** — templates or structures required?
5. **Existing patterns** — which sibling skill is the closest model?

If the answer to #3 is "nothing," you probably don't need a skill. A skill earns
its context cost by carrying knowledge the model lacks, not by restating what it
already does well.

## Frontmatter

```markdown
---
name: your-skill-name
description: What it does, and when to use it. Third person. Include trigger terms.
---
```

| Field | Rule |
| --- | --- |
| `name` | Max 64 chars, lowercase letters/numbers/hyphens, must match the directory name |
| `description` | Max 1024 chars, non-empty — this is how the agent decides to load the skill |

### The description is the whole discovery mechanism

Write it in **third person** (it gets injected into the system prompt), and
include both **what** and **when**:

- ✅ "Validate Core ML model numerical correctness against the PyTorch
  reference... Triggered by keywords like 'validate', 'numerical parity',
  'correlation', 'drift'."
- ❌ "Helps with models."
- ❌ "I can help you validate models."

Name the literal words a user would type. A skill with a vague description never
fires.

## Body structure

Follow the shape the existing skills use — it is consistent across this tree and
worth matching:

```markdown
# Skill Name

## Purpose
One paragraph. What question does this answer?

## Use When
Bullets. Concrete triggers.

## Do Not Use When
Bullets. Name the sibling skill that owns the adjacent case.

## Procedure
Numbered steps. Real commands with real paths.

## Anti-patterns
What goes wrong. This is often the highest-value section.

## Related skills
Where to hand off.
```

## Authoring principles

### 1. Concise is key

Context is shared with conversation history and every other skill. **Assume the
agent is smart.** Only add what it doesn't already have. Challenge every
paragraph: does this justify its token cost?

### 2. Under 500 lines

If SKILL.md is growing past that, move detail into `reference.md` or
`references/index.md` and link to it.

### 3. Progressive disclosure

Essentials in SKILL.md; detail in a sibling file the agent reads only when
needed. Keep references **one level deep** — `SKILL.md` → `reference.md`, not a
chain.

Two patterns already in use here:

- **`references/index.md`** — a map of canonical repo docs to read first
  (`create-plan`, `execute-plan`, `phase-audit`, `documentation`, `markdown`,
  `write-notes`).
- **`reference.md`** — an indexed reading list of repo paths
  (`ilya-sutskever`).

### 4. Match freedom to fragility

| Freedom | When | Example |
| --- | --- | --- |
| **High** (prose) | Many valid approaches | `david-ogilvy`, `ilya-sutskever` |
| **Medium** (templates) | Preferred pattern, some variation | `audit` output template |
| **Low** (exact commands) | Fragile, consistency critical | `bakeoff` measurement hygiene |

## Repo-specific rules for new skills

Any skill that touches verification **must** get these right, because they are
the two things a generic skill will get wrong here:

1. **There is no test suite.** No `test_*.py`, no `pyproject.toml`. A skill that
   says "run pytest" is wrong. The mechanical gate is:

   ```bash
   PYTHONPATH="$PWD" PYTORCH_ENABLE_MPS_FALLBACK=1 python3 train_CTC.py --epochs 1 --sanity
   ```

   See
   [The Mechanical Gate](../../../README/Skills/plan-workflow-skills-guide.md#the-mechanical-gate-repo-specific)
   for the per-surface table.

2. **CI is CPU-only Linux with torch + numpy.** A skill that recommends adding
   imports to the CTC sanity path must warn about
   [requirements-ci.txt](../../../requirements-ci.txt).

Also:

- **Git side effects need explicit authorization.** [CLAUDE.md](../../../CLAUDE.md)
  says only commit when asked; the exception is explicit invocation of a workflow
  skill. If your new skill commits or pushes, copy the Authority Model section
  from `execute-plan` and say so in the description.
- **Link with real relative paths** from the skill file
  (`../../../README/...`), and verify they resolve.
- **Don't duplicate a canonical doc.** Link to `README/Skills/`,
  `README/Guides/`, or `docs/` instead of restating it.

## Checklist before finishing

- [ ] `name` matches the directory name
- [ ] `description` is third person, names trigger words, states what **and** when
- [ ] "Do Not Use When" names the sibling skill that owns the adjacent case
- [ ] Every command is real and runnable in this repo
- [ ] No `pytest` claim unless test files actually exist
- [ ] Relative links resolve
- [ ] Under 500 lines, or detail moved to a reference file
- [ ] Git side effects, if any, have an explicit Authority Model

## Related skills

- [`markdown`](../markdown/SKILL.md) — repo markdown rules for the file you're
  writing.
- [`documentation`](../documentation/SKILL.md) — if the real task is inline code
  docs, not a skill.
