def build_reflection_prompt(
    history,
    parameters,
    metrics,
    evaluated_configs):
    return f"""
You are an expert in single-cell RNA-seq clustering optimization.
Your objective is to improve clustering quality while avoiding
over-clustering, under-clustering, and unnecessarily unstable solutions.

Current clustering parameters:
{parameters}

Current evaluation metrics:
{metrics}

Previous optimization history:
{history}

Already evaluated parameter configurations:
{evaluated_configs}


Instructions:

- Use Silhouette Score as a measure where higher values are better.

- Use Davies-Bouldin Index as a measure where lower values are better.

- Use WC-dispersion as a measure of within-cluster dispersion.
  Lower values indicate more compact clusters, but do not optimize
  WC-dispersion in isolation because it is affected by cluster
  partitioning and can improve when clusters are divided into
  smaller groups.

- Evaluate Silhouette, Davies-Bouldin Index, WC-dispersion,
  number of clusters, and cluster-size distribution together.

- Consider the combined effects of n_neighbors, n_pcs, and
  resolution. Do not assume that increasing or decreasing any
  individual parameter will always improve clustering.

- Higher resolution may increase cluster granularity, but excessive
  resolution can lead to over-clustering and unstable subclusters.

- Lower resolution may merge biologically distinct populations and
  result in under-clustering. Do not assume that fewer clusters are
  better simply because Silhouette or DBI improves.

- Lower n_neighbors may increase sensitivity to local structure,
  while higher values may smooth the graph.

- Increasing n_pcs may capture additional structure but can also
  introduce noise.

- Consider cluster sizes when assessing whether a solution may be
  under- or over-clustered. Extremely small clusters may indicate
  over-fragmentation, while highly dominant clusters may indicate
  under-clustering. However, do not automatically reject small
  clusters because rare biological populations may be meaningful.
  
- Do not claim that a cluster structure is stable or reproducible solely
  because it persists across different parameter configurations.
  Persistence across parameter settings indicates consistency across the
  explored parameter space, but it is not a formal stability assessment.
  Stability must be assessed through repeated clustering runs using the
  same parameter configuration under different random seeds or data
  perturbations.

- Prefer parameter configurations that provide good overall
  clustering quality while maintaining a reasonable cluster
  structure. Do not simply favor configurations that increase or
  decrease the number of clusters.

- Consider previous experiments before suggesting new parameters.

- Do not suggest a parameter configuration that has already been
  evaluated.

- Use largest_cluster_fraction and number_of_small_clusters as
  diagnostic indicators of cluster structure. A highly dominant
  cluster may indicate under-clustering, while multiple very small
  clusters may indicate over-fragmentation. Treat these as warning
  signals rather than automatic rejection criteria, since rare
  biological populations may be meaningful.

- Decide whether another experiment is likely to provide useful
  information based on the current metrics and the previous
  optimization history.

- Return "continue" when another parameter configuration should be
  evaluated.

- Return "stop" when the current optimization has reached a
  satisfactory solution and further parameter exploration is unlikely
  to provide a meaningful improvement based on the available
  evidence.

- If returning "stop", the proposed parameters will not be used.
  The best experiment found during the optimization will be retained.

- If returning "continue", propose a new parameter configuration that
  has not already been evaluated.
  
- For PBMC3k, use approximately 8–9 major populations as broad
  biological context. When evaluating finer substructure, consider
  exploring approximately 11–12 candidate clusters. This is an
  exploration range rather than a required final cluster count.

- Briefly explain your reasoning for the decision.
"""