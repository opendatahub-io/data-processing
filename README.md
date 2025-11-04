# Data Processing

![Status dev-preview](https://img.shields.io/badge/status-dev--preview-blue)
![GitHub License](https://img.shields.io/github/license/opendatahub-io/data-processing)
![GitHub Commits](https://img.shields.io/github/commit-activity/t/opendatahub-io/data-processing)

This repository provides reference **data-processing pipelines and examples** for [Open Data Hub](https://github.com/opendatahub-io) / [Red Hat OpenShift AI](https://www.redhat.com/en/products/ai/openshift-ai). It focuses on **document conversion** and **chunking** using the [Docling](https://docling-project.github.io/docling/) toolkit, packaged as [Kubeflow Pipelines](https://www.kubeflow.org/docs/components/pipelines/) (KFP), example [Jupyter Notebooks](https://jupyter.org/), and helper scripts.

The workbenches directory also provides a guide on how to create a custom [workbench image](https://github.com/opendatahub-io-contrib/workbench-images) to run Docling and the example notebooks in this repository.

## 📦 Repository Structure

```bash
data-processing
|
|- custom-workbench-image
|
|- kubeflow-pipelines
|   |- docling-standard
|   |- docling-vlm
|
|- notebooks
    |- tutorials
    |- use-cases
|
|- scripts
    |- subset_selection
```

## ✨ Getting Started

### Kubeflow Pipelines

Refer to the [Data Processing Kubeflow Pipelines](kubeflow-pipelines) documentation for instructions on how to install, run, and customize the [Standard](kubeflow-pipelines/docling-standard) and [VLM](kubeflow-pipelines/docling-vlm) pipelines.

### Notebooks

Data processing related [jupyter notebooks](notebooks) are broken down into [use-cases](notebooks/use-cases) and [tutorials](notebooks/tutorials). 

### Custom Workbench Image

Open Data Hub has the ability for users to add and run [custom workbench images](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/2.24/html/managing_openshift_ai/creating-custom-workbench-images).

A sample `Containerfile` and instructions to create a custom workbench image are in [custom-workbench-image](custom-workbench-image).

### Scripts

Curated scripts related to data processing live in this directory.

For example the [subset selection](scripts/subset_selection) scripts allows users to identify representative samples from large training datasets.

## 🤝 Contributing

We welcome issues and pull requests. Please:
- Open an issue describing the change.
- Include testing instructions.
- For pipeline/component changes, recompile the pipeline and update generated YAML if applicable.
- Keep parameter names and docs consistent between code and README.

## 📄 License

Apache License 2.0
