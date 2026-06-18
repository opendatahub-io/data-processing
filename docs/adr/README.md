# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records for the data-processing repository.

ADRs document significant technical decisions with their context, rationale, and consequences. They help contributors (and AI agents) understand *why* the code is structured the way it is.

## Index

| ADR | Title | Status |
|---|---|---|
| [001](001-kfp-component-serialization.md) | KFP Component Serialization | Accepted |
| [002](002-compiled-yaml-source-of-truth.md) | Compiled YAML as Source of Truth | Accepted |
| [003](003-shared-secret-name.md) | Shared Secret Name for S3 and VLM Configuration | Accepted |

## Creating a New ADR

1. Copy the template below
2. Name the file `NNN-short-title.md` (e.g., `004-new-decision.md`)
3. Fill in each section
4. Add an entry to the index table above
5. Submit as part of your PR

### Template

```markdown
# ADR-NNN: Title

## Status

Proposed | Accepted | Deprecated | Superseded by [ADR-NNN](NNN-title.md)

## Context

What is the issue that we're seeing that motivates this decision?

## Decision

What is the change that we're making?

## Consequences

What becomes easier or harder because of this change?
```
