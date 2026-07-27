# Documentation References

Canonical docs for `documentation`.

- `README/Guides/content/code-documentation-guide.md`
  Read first for the repo's actual inline code documentation philosophy and
  review checklist.
- `CLAUDE.md`
  Read for the LLM-first documentation rules — formal doc comments, explicit
  cross-file connections, named constants instead of magic numbers.
- `README/Mamba-on-Apple-Silicon.md`
  Read when the useful comment depends on MPS or ANE context. Code cites this
  guide **by section number**; keep those citations accurate.
- `docs/SYSTEM_ARCHITECTURE.md`
  Read when documenting a module boundary or data-flow contract.
- the most relevant `docs/...` guide for the subsystem you are documenting
  Read when the comment depends on export or app-integration context that is
  not obvious from the local file.
