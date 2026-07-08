# Agentic AI Framework for scRNA-seq Clustering Optimization

## Overview

This master's thesis investigates the use of an agentic AI framework to improve the robustness and reproducibility of clustering in single-cell RNA sequencing (scRNA-seq) analysis.

Rather than replacing established bioinformatics workflows, the project builds on the standard Scanpy pipeline by introducing an intelligent optimization layer that automatically evaluates and refines clustering parameters. The objective is to reduce manual parameter tuning while improving clustering quality through iterative evaluation and feedback.


## Motivation

Clustering results in scRNA-seq are highly sensitive to analysis parameters such as the number of principal components (PCs), number of neighbors, and Leiden clustering resolution. Small parameter changes can produce substantially different cluster structures and biological interpretations.

This project explores whether an AI-driven optimization agent can:

* Reduce manual trial-and-error during clustering
* Improve clustering robustness and reproducibility
* Systematically optimize clustering parameters
* Identify stable and high-quality clustering solutions using quantitative evaluation metrics


## Core Workflow

The framework consists of two main components:

1. A deterministic preprocessing pipeline using Scanpy (quality control, normalization, highly variable gene selection, PCA, and neighborhood graph construction)

2. An agentic optimization layer that:

   * explores different clustering parameter combinations,
   * evaluates clustering quality,
   * compares alternative solutions, and
   * recommends improved clustering configurations through iterative feedback.

## Methods and Tools

* Python
* Scanpy
* AnnData
* NumPy
* Pandas
* Matplotlib
* Jupyter Notebook
* LangGraph (agent orchestration)
* LangChain
* Scikit-learn

### Clustering Evaluation

The optimization agent evaluates clustering performance using metrics such as:

* Silhouette Score
* Davies–Bouldin Index
* Cluster size and structure
* Cluster stability across parameter configurations

## Datasets

* **PBMC 3K** – Baseline development and pipeline validation
* **PBMC 10K** – Clustering optimization and scalability evaluation
* **Human Pancreas Dataset** – Validation of the framework on an independent biological dataset
