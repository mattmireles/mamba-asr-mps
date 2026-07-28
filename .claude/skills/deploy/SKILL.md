---
name: deploy
description: Guard release decisions for mamba-asr-mps. Use when packaging, publishing, or claiming a production-ready export. The current WER 0.612790 fails the 0.25 accuracy gate, so deployment is blocked unless a later preregistered and recorded result supersedes it.
---

# Deploy

Current status: **blocked by accuracy**.

Before any release:

1. Confirm a newer frozen evaluation explicitly supersedes WER `0.612790`.
2. Require WER `<= 0.25` and all numerical/state validation gates.
3. Run the complete package, install, smoke, and artifact-hash checks.
4. Publish only the claims proven by those artifacts.

Do not ship or advance phases because packaging succeeds while accuracy fails.
