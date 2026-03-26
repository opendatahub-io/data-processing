# ADR-003: Shared Secret Name for S3 and VLM Configuration

## Status

Accepted

## Context

KFP pipelines need credentials for two optional features:
1. **S3 access** — downloading PDFs from S3-compatible storage (`import_pdfs` component)
2. **Remote VLM access** — calling a remote vision-language model API (`docling_convert_vlm` component, VLM pipeline only)

These could use separate Kubernetes Secrets (e.g., `data-processing-s3` and `data-processing-vlm`) or a single shared secret.

## Decision

Both features use a single Kubernetes Secret named `data-processing-docling-pipeline`, mounted at `/mnt/secrets`. The secret is mounted as `optional: true` so pipelines work without it when using HTTP sources and local models.

S3 keys: `S3_ENDPOINT_URL`, `S3_ACCESS_KEY`, `S3_SECRET_KEY`, `S3_BUCKET`, `S3_PREFIX`
VLM keys: `REMOTE_MODEL_ENDPOINT_URL`, `REMOTE_MODEL_API_KEY`, `REMOTE_MODEL_NAME`

Components validate only the keys they need — `import_pdfs` checks for S3 keys when `from_s3=True`, and `docling_convert_vlm` checks for VLM keys when `remote_model_enabled=True`.

## Consequences

- **Positive**: Users manage one secret instead of two — simpler setup
- **Positive**: Single `kubernetes.use_secret_as_volume()` call per component
- **Positive**: Keys have distinct prefixes so there's no collision
- **Negative**: The secret can grow large if both S3 and VLM are configured
- **Negative**: Changing S3 credentials requires updating a secret that also contains VLM credentials (and vice versa)
- **Negative**: RBAC cannot be scoped separately to S3 vs VLM access
