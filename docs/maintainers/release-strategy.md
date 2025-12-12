# Data Processing Release Strategy

This document outlines the release approach for the ODH Data Processing repository, which provides Kubeflow Pipeline components and Jupyter notebooks for document processing workflows.

## Initial Release (1.0.0)

The first release of the Data Processing hub will be **version 1.0.0**, establishing the foundational pipeline components and notebook examples.  Following [Semantic Versioning](https://semver.org/), future releases will use `X.Y.Z` numbering where major versions (X) indicate breaking changes, minor versions (Y) introduce new features and components, and patch versions (Z) provide bug fixes and security updates.

## Release Approach

The project will follow an **adhoc release schedule** based on documentation needs and significant feature additions rather than fixed timelines. Each release will be tagged as `vX.Y.Z` and serve as a stable snapshot that users can reference. The `main` branch serves as the development branch with releases created from stable snapshots when ready for external reference.
