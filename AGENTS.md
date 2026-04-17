# Iolex Project Guidelines

## Agent Usage
- Use the architectural expert for project-wide understanding, minimal-change planning, and cross-file design.
- Use the code builder for all file edits and implementation work.
- Use the reviewer/refactorer for hard-nosed critique and correctness checks.
- Use the theory expert when the conceptual or statistical validity of an approach is uncertain.
- Use the practice expert for PyTorch, vectorization, performance, memory, and computational efficiency.

## Code Style
- Prefer the smallest change that solves the problem.
- Keep logic explicit; do not add silent exception swallowing.
- Avoid unnecessary boilerplate and indirection.
- Preserve the current style of the touched code.

## Validation
- Run only the checks that are relevant to the change.
- Fix real errors at the source rather than patching around them.
