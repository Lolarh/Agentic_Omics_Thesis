# Agentic AI for scRNA-seq Clustering Optimization

## Overview

This master’s thesis project focuses on developing improving and automating clustering decisions in single-cell RNA sequencing (scRNA-seq) analysis.

The goal is not to replace existing bioinformatics tools, but to support them by making clustering workflows more robust, and reproducible.

The system builds on established tools such as Scanpy and investigates how an agent-based approach can help improve clustering quality in a structured and automated way.

---

## Motivation

In scRNA-seq analysis, clustering results are highly sensitive to parameter choices (for example; number of neighbors, number of principal components, clustering resolution). Small changes can lead to significantly different biological interpretations.

This project explores whether an AI-assisted system can:
- reduce manual trial-and-error in clustering
- improve robustness of clustering outcomes
- support more systematic parameter selection
- help identify unstable or unreliable cluster configurations

---

## Core Idea

The system will:

1. Load single-cell RNA-seq data (initially PBMC datasets)
2. Run a standard Scanpy pipeline (preprocessing → clustering → visualization)
3. Test different clustering parameter configurations
4. Evaluate clustering results using simple quantitative and structural metrics
5. Compare results to identify more stable or meaningful cluster structures

## Methods and Tools

- Python
- Scanpy
- AnnData
- Snakemake
- Jupyter Notebooks
- NumPy / Pandas / Matplotlib
- Custom clustering optimization agent: System that evaluates multiple clustering configurations using metrics such as silhouette score, cluster stability, and marker gene enrichment to select optimal solutions

---

## Dataset
PBMC 10k: Primary dataset to develop and optimize clustering pipeline and agent

Human pancreas dataset: Validation dataset for testing generalization across biological systems