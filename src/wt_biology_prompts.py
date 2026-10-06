MURARO_BIOLOGICAL_CONTEXT = """
DATASET-SPECIFIC BIOLOGICAL CONTEXT:

The dataset is the Muraro human pancreas single-cell RNA-seq
dataset.

The dataset contains heterogeneous pancreatic cell populations
spanning multiple broad biological compartments.

Broad knowledge of pancreatic biology may be used only as contextual
information when interpreting the supplied marker-gene patterns.

Do not use a predefined list of pancreatic cell types as optimization
targets.

Do not assume that every known pancreatic population must appear as a
separate cluster.

Do not force the clustering toward a predetermined number of
clusters or biological populations.

Do not infer the identity of an individual cluster unless the
supplied marker-gene pattern provides reasonable supporting evidence.

The biological context does not establish the identity of any
individual cluster and must not be treated as ground-truth
annotation.
"""


PBMC3K_BIOLOGICAL_CONTEXT = """
DATASET-SPECIFIC BIOLOGICAL CONTEXT:

The dataset is the PBMC3K single-cell RNA-seq dataset.

Broad knowledge of PBMC biology may be used only as contextual
information when interpreting the supplied marker-gene patterns.

Do not use a predefined list of expected cell types as optimization
targets.

Do not assume that every known PBMC population must appear as a
separate cluster.

Do not force the clustering toward a predetermined number of
clusters or biological populations.

Do not infer the identity of an individual cluster unless the
supplied marker-gene pattern provides reasonable supporting evidence.

The biological context does not establish the identity of any
individual cluster and must not be treated as ground-truth
annotation.
"""


def build_biology_reflection_prompt(
    history,
    parameters,
    metrics,
    evaluated_configs,
    max_iterations,
    dataset_context,
):
    return f"""
You are an expert in single-cell RNA-seq clustering optimization.

Your objective is to explore clustering parameters and identify
parameter configurations that provide strong overall clustering
evidence while avoiding unnecessary over-fragmentation or
under-clustering.

The optimization should balance potential improvement against the
cost of additional experiments.

BIOLOGY-INFORMED EXPERIMENTAL CONDITION:

This is a biology-informed clustering optimization experiment.

The optimization uses three sources of information:

1. intrinsic clustering metrics,
2. cluster structure, and
3. limited biological evidence from marker genes.

These sources of information have different roles.

Intrinsic clustering metrics are the primary quantitative evidence.

Cluster structure is secondary structural evidence used to interpret
the clustering solution.

Marker genes are secondary biological evidence used only to provide
context for the observed clustering.

Biological evidence must never be treated as ground-truth annotation
or as a predefined target for the optimization.

Do not force the clustering toward a predetermined number of
clusters, cell types, lineages, or expected biological outcome.

DATASET-SPECIFIC BIOLOGICAL CONTEXT:

{dataset_context}

Use the dataset-specific context only as broad background for
interpreting the supplied marker-gene patterns.

Do not treat this context as evidence that a particular population
must exist.

Do not use this context to determine the desired number of clusters.

Do not use this context to select a parameter configuration in the
absence of supporting intrinsic or structural evidence.

CURRENT EXPERIMENT:

Current clustering parameters:
{parameters}

Current evaluation metrics:
{metrics}

Previous optimization history:
{history}

Already evaluated parameter configurations:
{evaluated_configs}

EVALUATION PRINCIPLES:

Use the intrinsic metrics together rather than optimizing any single
metric in isolation.

* Silhouette Score provides evidence about cluster separation and
  cohesion. Higher values are generally favorable.

* Davies-Bouldin Index provides additional evidence about cluster
  separation and cohesion. Lower values are generally favorable.

* WC-dispersion provides evidence about within-cluster dispersion.

* Banfield-Raftery provides an additional intrinsic clustering
  criterion.

Metric values must be interpreted relative to the other evaluated
configurations rather than in isolation.

When metrics disagree, explicitly acknowledge the trade-off.

Do not assume that improvement in one metric necessarily means that
the overall clustering solution has improved.

CLUSTER STRUCTURE:

Consider cluster structure as contextual evidence alongside the
intrinsic metrics.

* The number of clusters represents clustering granularity, not a
  target.

* Higher resolution may increase granularity but may also introduce
  unnecessary fragmentation.

* Lower resolution may merge structure but fewer clusters are not
  automatically preferable.

* Small clusters are not automatically artifacts.

* Small clusters are also not automatically biologically meaningful.

* A dominant cluster may indicate possible under-clustering, but
  dominance alone is not sufficient evidence to reject a solution.

* Consider the complete cluster-size distribution rather than focusing
  only on the smallest or largest cluster.

* Interpret structural changes together with changes in intrinsic
  metrics.

BIOLOGICAL EVIDENCE:

The evaluation may include a small number of top-ranked marker genes
for each cluster:

{metrics.get("marker_genes", {})}

Marker genes are supporting biological evidence only.

IMPORTANT:

The marker genes are derived from the clustering solution currently
being evaluated.

Therefore, they describe characteristics of the resulting clusters
and must not be treated as independent validation of those clusters.

Use marker genes cautiously to assess whether the observed clustering
appears broadly compatible with plausible biological structure.

Do not treat marker genes as cell-type annotations.

Do not assign definitive cell-type identities based only on the
supplied marker genes.

Do not infer biological identity from a single marker gene.

Consider patterns across multiple genes when interpreting biological
evidence.

Do not assume that shared markers mean that two clusters represent
the same biological population.

Do not assume that different marker genes prove that two clusters are
biologically distinct.

Do not describe clusters as duplicated, redundant, or biologically
identical unless the available evidence strongly supports that
interpretation.

Do not describe a cluster as a specific biological population unless
the supplied marker pattern provides reasonable support.

Use cautious language such as:

* "consistent with"
* "suggestive of"
* "compatible with"
* "may reflect"
* "uncertain"

If the biological interpretation is ambiguous, explicitly state that
it is ambiguous.

Technical, stress-related, proliferative, mitochondrial,
housekeeping, spike-in, or other state-associated signals may
contribute to marker patterns.

Do not interpret such signals as independent biological populations
without additional supporting evidence.

Do not invent biological explanations that are not supported by the
supplied marker genes.

BIOLOGICAL EVIDENCE MUST REMAIN SECONDARY:

Biological evidence must not independently determine the optimization
decision.

* Do not recommend a parameter configuration solely because its
  marker genes appear biologically interesting.

* Do not reject a configuration solely because a cluster is small.

* Do not prefer a configuration solely because it appears to contain
  a recognizable biological population.

* Do not use biological plausibility to compensate for clear
  deterioration in the intrinsic clustering evidence.

* Biological evidence may support, weaken, or leave unchanged an
  interpretation based on intrinsic metrics and cluster structure.

* When biological and intrinsic evidence disagree, explicitly state
  the disagreement.

* When biological evidence is ambiguous, do not resolve the
  ambiguity by assuming a biological explanation.

PARAMETER REASONING:

Consider the combined effects of n_neighbors, n_pcs, and resolution.

Do not assume that changing one parameter will always improve
clustering.

Consider higher resolution when additional partitioning is supported
by improvement or meaningful trade-offs in intrinsic metrics and
reasonable cluster structure.

Be cautious about increasing resolution when the main effect is an
increase in cluster number without meaningful improvement in
intrinsic evidence.

Be cautious about decreasing resolution when intrinsic or structural
evidence suggests that meaningful structure may be excessively
merged.

Biological evidence may provide additional context for a parameter
change, but it must not be the sole justification for that change.

When selecting a new configuration, prefer an experiment that tests a
specific uncertainty identified from the previous results rather than
making multiple arbitrary parameter changes.

PARAMETER CONSTRAINTS:

* n_neighbors must be between 10 and 50.

* n_pcs must be between 10 and 50.

* resolution must be between 0.1 and 2.0.

* Do not propose parameter values outside these ranges.

* Do not propose a configuration that has already been evaluated.

STABILITY:

Do not claim that a clustering solution is formally stable or
reproducible unless stability has explicitly been assessed through
repeated clustering under different random seeds or data
perturbations.

EXPLORATION BUDGET:

The optimization has a maximum exploration budget defined by the
agent.

The maximum budget is a limit, not a requirement to use all available
experiments.

Stop when the available evidence is sufficient or when further
exploration is unlikely to provide meaningful additional information.

DECISION:

Return "continue" when another unevaluated configuration is likely to
provide useful information about the optimization landscape or
meaningfully improve the clustering solution.

Return "stop" when the available evidence is sufficient or further
exploration is unlikely to provide meaningful additional information.

When continuing, propose exactly one new, unevaluated configuration.

OUTPUT REQUIREMENTS:

Return:

1. decision:
   "continue" or "stop"

2. reason:
   A concise explanation based primarily on intrinsic metrics and
   cluster structure, with biological evidence used only as
   supporting context.

3. parameters:
   If continuing, provide one valid unevaluated configuration.

The reasoning should distinguish clearly between:

* intrinsic evidence,
* structural evidence, and
* biological evidence.

Do not present uncertain biological interpretations as established
facts.

Do not invent biological explanations unsupported by the supplied
marker-gene evidence.
"""

    
# ============================================================
# BIOLOGY-INFORMED FINAL SELECTION PROMPT
# ============================================================

def build_biology_final_selection_prompt(
    history,
    dataset_context,
):
    return f"""
You are an expert in single-cell RNA-seq clustering evaluation.

Your task is to select the preferred clustering experiment from the
completed experiment history.

This is a biology-informed experimental condition.

Biological information is intentionally limited and must not be
treated as ground-truth annotation.

DATASET-SPECIFIC BIOLOGICAL CONTEXT:

{dataset_context}

Use this context only as broad background for interpreting the
supplied marker-gene evidence.

Do not use it as a predefined target for the number of clusters,
cell types, or biological populations.

COMPLETED EXPERIMENT HISTORY:

{history}

FINAL SELECTION PRINCIPLES:

Select exactly one experiment that already exists in the completed
experiment history.

The selected_iteration must exactly match the iteration number of one
completed experiment.

Do not propose new clustering parameters.

Evaluate the completed experiments using the following hierarchy:

1. Intrinsic clustering evidence.
2. Cluster structure.
3. Biological evidence from marker genes.

Biological evidence is supporting context and must not independently
determine the final selection.

INTRINSIC CLUSTERING EVIDENCE:

Consider the intrinsic metrics together.

* Silhouette Score: higher values are generally favorable.

* Davies-Bouldin Index: lower values are generally favorable.

* WC-dispersion: lower values generally indicate lower
  within-cluster dispersion.

* Banfield-Raftery: use as an additional intrinsic criterion.

Do not select an experiment based on a single metric.

Explicitly acknowledge important metric trade-offs.

Prefer the configuration with the strongest overall intrinsic
evidence rather than automatically selecting the configuration with
the best value for one metric.

CLUSTER STRUCTURE:

Consider the number of clusters and complete cluster-size
distribution.

* Do not assume fewer clusters are better.

* Do not assume more clusters are better.

* Small clusters are not automatically artifacts.

* Small clusters are not automatically biologically meaningful.

* A dominant cluster may indicate possible under-clustering, but
  should not determine the selection by itself.

* Consider whether changes in resolution or graph parameters produce
  meaningful structural changes relative to changes in intrinsic
  metrics.

BIOLOGICAL EVIDENCE:

The experiment history may contain a small number of top-ranked
marker genes for each cluster.

Marker genes must be treated as secondary biological evidence.

Importantly, these marker genes are derived from the corresponding
clustering solutions.

They therefore describe the resulting clusters and should not be
treated as independent validation or ground-truth annotation.

* Do not assign definitive cell-type identities based only on marker
  genes.

* Do not infer biological identity from a single marker.

* Interpret marker patterns cautiously and in groups.

* Do not treat shared markers as automatic evidence that two clusters
  are the same population.

* Do not treat different markers as proof that two clusters are
  biologically distinct.

* Do not describe clusters as duplicated or redundant solely from
  overlapping marker genes.

* Do not treat small clusters as biologically meaningful solely
  because recognizable markers are present.

* Do not treat small clusters as artifacts solely because they are
  small.

* Technical, stress-related, proliferative, mitochondrial,
  housekeeping, spike-in, or other state-associated signals should
  not by themselves be interpreted as distinct biological
  populations.

Use cautious language such as "consistent with", "suggestive of",
"compatible with", or "uncertain".

If biological evidence is ambiguous, state that it is ambiguous.

OVER-FRAGMENTATION:

A higher-resolution experiment should not be preferred simply because
it produces more biologically interpretable clusters.

A lower-resolution experiment should not be preferred simply because
it produces fewer clusters.

Additional clusters may be beneficial when supported by improved
intrinsic evidence and reasonable cluster structure.

If additional clusters provide little intrinsic improvement and
mainly introduce very small or weakly supported groups, this may
provide evidence for over-fragmentation.

Do not use biological marker patterns to manufacture a justification
for additional clusters.

OVERALL COMPARISON:

Evaluate the experiments in this order:

1. Compare intrinsic clustering evidence.

2. Compare cluster structure and potential fragmentation.

3. Determine whether marker-gene evidence provides:
   - supporting evidence,
   - contradictory evidence, or
   - no meaningful additional evidence.

Biological evidence does not have to agree with the intrinsic
metric winner.

If biological evidence conflicts with intrinsic evidence, describe
the conflict explicitly.

Do not allow biological plausibility alone to overturn substantially
stronger intrinsic evidence.

If the intrinsic-metric winner also has reasonable cluster structure
and there is no strong contradictory evidence, prefer it.

FINAL DECISION:

Select exactly one completed experiment.

In the reasoning, explicitly state:

1. Which experiment has the strongest overall intrinsic evidence.

2. Which experiment has the most reasonable cluster structure.

3. Whether the biological evidence provides:
   - supporting evidence,
   - contradictory evidence, or
   - no meaningful additional evidence.

4. Which experiment is ultimately selected.

5. If the final selection differs from the strongest intrinsic
   candidate, explain the structural evidence or other strong evidence
   that justifies the difference.

Do not select an experiment solely because it has:

* the highest Silhouette Score;
* the lowest Davies-Bouldin Index;
* the lowest WC-dispersion;
* the lowest Banfield-Raftery score;
* the fewest clusters;
* the most clusters; or
* the most recognizable biological marker genes.

Return only one completed iteration as the final selection.
"""