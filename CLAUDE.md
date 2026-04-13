# CLAUDE.md

## Engineering Practices

- **TDD with strict red-green-refactor.** Write a failing test first, make it
  pass with the simplest code, then refactor. Do not skip the refactor step —
  genuinely pause and critically evaluate what could be cleaner before moving on.
- **Clean separation of concerns.** The crate-level architecture enforces this
  at the large scale, but it applies equally within crates — modules, structs,
  and functions should each have a single clear responsibility.
- **Loose coupling.** Depend on traits and interfaces, not concrete
  implementations. This applies at every scale, not just between crates.
- **Human-readable code.** No single-letter variable names. Names should convey
  intent.
- **Small functions.** Functions must be at most 50 lines after `cargo fmt`.
- **Modular structure.** Prefer many small, focused modules over large files.
